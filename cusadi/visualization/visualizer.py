from typing import Dict
from pathlib import Path
from io import BytesIO
import viser
import numpy as np
from viser.extras import ViserUrdf
import yourdfpy
import time
import subprocess
import trimesh
from PIL import Image
from cusadi.utils.coord_conventions \
    import convert_coordinates, convert_floating_base, convert_joint_order
from cusadi.utils.coord_conventions import VISER_CONVENTION

class Visualizer3D:
    urdf_instances = {}
    meshes = {}
    trajectory_folders = {}
    server: viser.ViserServer

    def __init__(self):
        self.server = viser.ViserServer()
        self.setup_scene()
        self.setup_lighting()
        self.setup_camera()

    def setup_scene(self):
        grid = self.server.scene.add_grid(
            "/grid", width=100, height=100, position=(0.0, 0.0, 0.0))
        grid.shadow_opacity = 0.6

    def setup_lighting(self):
        # Lighting setup
        self.server.scene.configure_default_lights(enabled=False, cast_shadow=False)
        gui_default_lights = self.server.gui.add_checkbox("Default lights", initial_value=False)
        gui_default_shadows = self.server.gui.add_checkbox("Default shadows", initial_value=False)
        gui_default_lights.on_update(
            lambda _: self.server.scene.configure_default_lights(
                gui_default_lights.value, gui_default_shadows.value))
        gui_default_shadows.on_update(
            lambda _: self.server.scene.configure_default_lights(
                gui_default_lights.value, gui_default_shadows.value))
        self.server.scene.add_transform_controls(
            "/lighting", position=(-1.0, 0.0, 2.0), scale=1.0)
        self.server.scene.add_label("/lighting/label", "Directional")
        directional_light = self.server.scene.add_light_directional(
            name="/lighting/directional_light",
            color=(255, 255, 255),
            cast_shadow=True,
            position=(0, 0, 2.0))
        # Create light control inputs.
        with self.server.gui.add_folder("Directional light"):
            gui_directional_color = self.server.gui.add_rgb("Color", initial_value=directional_light.color)
            gui_directional_intensity = self.server.gui.add_slider(
                "Intensity",
                min=0.0,
                max=20.0,
                step=0.01,
                initial_value=2,
            )
            gui_directional_shadows = self.server.gui.add_checkbox("Shadows", True)
            @gui_directional_color.on_update
            def _(_) -> None:
                directional_light.color = gui_directional_color.value
            @gui_directional_intensity.on_update
            def _(_) -> None:
                directional_light.intensity = gui_directional_intensity.value
            @gui_directional_shadows.on_update
            def _(_) -> None:
                directional_light.cast_shadow = gui_directional_shadows.value

    def setup_camera(self):
        button_camera_fov = self.server.gui.add_button_group(
            "FOV", ("Rectilinear", "Perspective"))
        @button_camera_fov.on_click
        def _(event: viser.GuiEvent) -> None:
            client = event.client
            if button_camera_fov.value == "Rectilinear":
                client.camera.fov = np.pi/20
            elif button_camera_fov.value == "Perspective":
                client.camera.fov = 1.3

    def get_server(self):
        return self.server
    
    def get_client(self):
        clients = self.server.get_clients()
        client_ids = list(clients.keys())
        if len(client_ids) == 0:
            print("No clients connected!")
            return None
        return clients[client_ids[0]]  # Return the first connected client
        
    def remove_all(self):
        self.remove_all_urdfs()
        self.remove_all_meshes()
        self.remove_all_trajectories()

    ########### URDF functionality ###########
    def get_urdf(self, urdf_filepath):
        urdf = yourdfpy.URDF.load(urdf_filepath)
        return urdf

    def add_urdf(self, urdf, instance_name,
                 src_convention=VISER_CONVENTION, mesh_rgba=None):
        # Check if urdf is filepath or URDF object
        if isinstance(urdf, str):
            urdf = self.get_urdf(urdf)
        base = self.server.scene.add_frame('/urdf/' + instance_name, show_axes=False)
        viser_urdf = ViserUrdf(self.server, urdf,
                            root_node_name=base.name,
                            mesh_color_override=mesh_rgba)
        urdf_convention = VISER_CONVENTION
        urdf_convention.set_joint_order(list(viser_urdf.get_actuated_joint_names()))
        self.urdf_instances[instance_name] = {'urdf': viser_urdf,
                                              'base': base,
                                              'src_convention': src_convention,
                                              'viser_convention': urdf_convention}
        print(f"Added URDF instance '{instance_name}'")

    def update_urdf(self, name, pos, ori, q):
        viser_urdf = self.urdf_instances[name]['urdf']
        base = self.urdf_instances[name]['base']
        if viser_urdf is None:
            print(f"URDF instance '{name}' not found!")
            return
        if pos is not None:
            base.position = np.array(pos) if pos is not np.array else pos
        if ori is not None:
            if ori.ndim == 2:# Rotation matrix
                base_SO3 = viser.transforms.SO3.from_matrix(ori)
                base.wxyz = base_SO3.wxyz
            elif len(ori) == 3: # RPY convention
                base_SO3 = viser.transforms.SO3.from_rpy_radians(q[3], q[4], q[5])
                base.wxyz = base_SO3.wxyz
            elif len(ori) == 4: # Quat
                base.wxyz = np.array([ori[3], ori[0], ori[1], ori[2]])
        viser_urdf.update_cfg(q)

    def remove_urdf(self, name):
        if name not in self.urdf_instances:
            print(f"URDF instance '{name}' not found!")
            return
        viser_urdf = self.urdf_instances[name]['urdf']
        viser_urdf.remove()
        del self.urdf_instances[name]
        print(f"Removed URDF instance '{name}'")
    
    def remove_all_urdfs(self):
        for name in list(self.urdf_instances.keys()):
            self.remove_urdf(name)

    def remove_trajectory(self, name):
        if name not in self.trajectory_folders:
            print(f"Trajectory folder '{name}' not found!")
            return
        folder = self.trajectory_folders[name]
        folder.remove()
        del self.trajectory_folders[name]
        print(f"Removed trajectory folder '{name}'")
    
    def remove_all_trajectories(self):
        for name in list(self.trajectory_folders.keys()):
            self.remove_trajectory(name)

    ########### Animation/recording functionality ###########

    def add_trajectory(self, label, handles, p_traj, dt=0.01):
        button_play = self.server.gui.add_button(f"Play {label}")
        button_step = self.server.gui.add_button_group("Frame", ("Prev", "Next"))
        slider_trajectory = self.server.gui.add_slider(
            f"Timestep", min=0, max=p_traj.shape[1], step=1, initial_value=0)
        @button_play.on_click
        def _(_) -> None:
            for i in range(p_traj.shape[1]):
                for h, p in zip(handles, p_traj[:, i]):
                    h.position = p
                slider_trajectory.value = i
                time.sleep(dt)
        @button_step.on_click
        def _(_) -> None:
            if button_step.value == "Prev":
                slider_trajectory.value -= 1
            elif button_step.value == "Next":
                slider_trajectory.value += 1
        @slider_trajectory.on_update
        def _(_):
            i = slider_trajectory.value
            for h, p in zip(handles, p_traj[:, i]):
                h.position = p

    def add_urdf_trajectory(self, name, p_traj, ori_traj, jnt_traj, dt=0.01):
        if name not in self.urdf_instances:
            print(f"URDF instance '{name}' not found!")
            return
        # TODO: slider for trajectory
        # button_play_animation = self.server.gui.add_button("Play trajectory")
        
        self.trajectory_folders[name] = self.server.gui.add_folder(name + " trajectory")
        with self.trajectory_folders[name]:
            button_play = self.server.gui.add_button("Play trajectory")
            button_step = self.server.gui.add_button_group("Frame", ("Prev", "Next"))
            slider_trajectory = self.server.gui.add_slider(
                f"Timestep", min=0, max=p_traj.shape[1], step=1, initial_value=0)
            
            is_playing = [False]
            @button_play.on_click
            def _(_) -> None:
                is_playing[0] = True
                t_start = time.perf_counter()
                for i in range(p_traj.shape[1]):
                    with self.server.atomic():
                        self.update_urdf(name, p_traj[:, i], ori_traj[:, i], jnt_traj[:, i])
                    slider_trajectory.value = i
                    # Sleep only for the remaining time until the next frame deadline
                    deadline = t_start + (i + 1) * dt
                    remaining = deadline - time.perf_counter()
                    if remaining > 0:
                        time.sleep(remaining)
                is_playing[0] = False


            @slider_trajectory.on_update
            def _(_):
                if is_playing[0]:
                    return
                i = slider_trajectory.value
                with self.server.atomic():
                    self.update_urdf(name, p_traj[:, i], ori_traj[:, i], jnt_traj[:, i])

            # @button_play.on_click
            # def _(_) -> None:
            #     for i in range(p_traj.shape[1]):
            #         with self.server.atomic():
            #             self.update_urdf(name, p_traj[:, i], ori_traj[:, i], jnt_traj[:, i])
            #             slider_trajectory.value = i
            #         time.sleep(dt)
            @button_step.on_click
            def _(_) -> None:
                if button_step.value == "Prev":
                    slider_trajectory.value -= 1
                elif button_step.value == "Next":
                    slider_trajectory.value += 1

            self.button_record_animation = self.server.gui.add_button("Record trajectory")
            self.text_record_foldername = self.server.gui.add_text(
                "Output folder", initial_value="animation_frames")
            self.text_record_filename = self.server.gui.add_text(
                "Filename", initial_value="animation")
            self.camera_tracking = self.server.gui.add_checkbox(
                "Track", initial_value=False)
            self.camera_offset = self.server.gui.add_vector3(
                "Cam. offset", initial_value=(1.0, -12.0, 1.0))
            
            @self.button_record_animation.on_click
            def _(event: viser.GuiEvent) -> None:
                target_fps = 30
                input_fps = 1.0 / dt  # original frame rate of your trajectory
                frame_skip = max(1, int(input_fps / target_fps))
                client = event.client
                assert client is not None
                output_dir = self.text_record_foldername.value
                filename = self.text_record_filename.value
                frame_idx = 0
                for i in range(0, jnt_traj.shape[1], frame_skip):
                    self.update_urdf(name, p_traj[:, i], ori_traj[:, i], jnt_traj[:, i])
                    frame = client.camera.get_render(height=1080, width=1920,
                                                    transport_format='png')
                    self.save_frame(frame, output_dir=output_dir, frame_idx=frame_idx)
                    frame_idx += 1
                print(f"Saved {frame_idx} frames to '{output_dir}'.")
                output_video = f"{output_dir}/{filename}.mp4"
                self.frames_to_video(output_dir, output_video, fps=target_fps)
                self.delete_frames(output_dir)
                print(f"Created video '{output_video}' and removed '{output_dir}'.")

    def position_camera(self, position):
        client = self.get_client()
        if client is None:
            return
        client.camera.fov = np.pi/20
        client.camera.position = position

    def get_trajectory_frames(self, name, trajectory):
        frames = []
        client = self.get_client()

        camera_pos_prev = client.camera.position
        for i in range(trajectory.shape[1]):
            if self.camera_tracking.value:
                camera_rel_traj = trajectory[0:3, i] + self.camera_offset.value
                camera_xyz = 0.2*camera_rel_traj + 0.8*camera_pos_prev
                self.position_camera(position=camera_xyz)
                if i == 0:
                    client.camera.position = camera_rel_traj
                    client.camera.look_at = trajectory[:3, i]
                camera_pos_prev = camera_xyz
            self.update_urdf(name, trajectory[:, i])
            frame = client.get_render(height=1080, width=1920)
            frames.append(frame)
        return frames

    def save_frame(self, frame, output_dir='animation_frames', frame_idx=0):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        frame_path = output_path / f"frame_{frame_idx:06d}.png"

        if isinstance(frame, (bytes, bytearray)):
            with Image.open(BytesIO(frame)) as img:
                img.save(frame_path)
            return

        frame_array = np.asarray(frame, dtype=np.uint8)
        if frame_array.ndim != 3:
            raise ValueError("Frames must be PNG bytes or HWC RGB/RGBA arrays.")
        if frame_array.shape[2] == 3:
            Image.fromarray(frame_array, mode="RGB").save(frame_path)
            return
        if frame_array.shape[2] == 4:
            Image.fromarray(frame_array, mode="RGBA").save(frame_path)
            return
        raise ValueError("Frames must be PNG bytes or HWC RGB/RGBA arrays.")

    def frames_to_video(self, frames_dir, output_filename, fps=30):
        frames_path = Path(frames_dir)
        command = [
            "ffmpeg",
            "-loglevel", "error",
            "-y",
            "-framerate", str(fps),
            "-i", str(frames_path / "frame_%06d.png"),
            "-filter_complex",
            "[0:v]format=rgba,split[fg][ref];[ref]geq=r=255:g=255:b=255:a=255[bg];[bg][fg]overlay,format=yuv420p",
            "-c:v", "libx264",
            output_filename,
        ]
        result = subprocess.run(command)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed while writing '{output_filename}'.")

    def delete_frames(self, frames_dir):
        frames_path = Path(frames_dir)
        for frame_path in frames_path.glob("frame_*.png"):
            frame_path.unlink()
        if not any(frames_path.iterdir()):
            frames_path.rmdir()

    def compress_video(self, video_filepath):
        # Use ffmpeg to compress the video
        compressed_filepath = video_filepath.replace('.mp4', '_compressed.mp4')
        command = [
            'ffmpeg',
            '-loglevel', 'error',  # Suppress output except for errors
            '-i', video_filepath,
            '-vcodec', 'libx265',
            '-crf', '28',  # Adjust CRF value for quality (lower is better quality)
            '-y', compressed_filepath
        ]
        subprocess.run(command)
        print(f"Compressed video saved as '{compressed_filepath}'")
        return compressed_filepath

    ########### Mesh functionality ###########

    # Smoothed surface
    def add_simple_mesh(self, name, mesh: Dict,
                        position=(0, 0, 0), color=(66, 51, 31)):
        vertices = mesh['vertices']
        faces = mesh['faces']
        viser_mesh = self.server.scene.add_mesh_simple('/' + name,
                                          vertices=vertices,
                                          faces=faces,
                                          position=position,
                                          color=color)
        self.meshes[name] = viser_mesh

    # Discrete polygons
    def add_trimesh(self, name, trimesh: trimesh.Trimesh,
                    position=(0, 0, 0)):
        viser_mesh = self.server.scene.add_mesh_trimesh('/' + name,
                                           mesh=trimesh,
                                           position=position)
        self.meshes[name] = viser_mesh
    
    def remove_mesh(self, name):
        if name not in self.meshes:
            print(f"Mesh '{name}' not found!")
            return
        viser_mesh = self.meshes[name]
        viser_mesh.remove()
        del self.meshes[name]
        print(f"Removed mesh '{name}'")
    
    def remove_all_meshes(self):
        for name in list(self.meshes.keys()):
            self.remove_mesh(name)

    ########## Visual debugging functionality ##########
    def add_debug_sphere(self, name, position=(0,0,0),
                         radius=0.02, color=(255,0,0)):
        sphere = self.server.scene.add_icosphere(
            '/debug/' + name, position=position, radius=radius, color=color)
        return sphere

    def add_debug_frame(self, name, position=(0,0,0)):
        mesh_handle = self.server.scene.add_frame('/' + name,
                                          position=position,
                                          wxyz=(1, 0, 0, 0),
                                          axes_length=0.2,
                                          axes_radius=0.005)
        return mesh_handle

    def add_frame_trajectory(self, label, handle, traj, dt=0.01):
        button_play = self.server.gui.add_button(f"Play {label}")
        button_step = self.server.gui.add_button_group("Frame", ("Prev", "Next"))
        slider_trajectory = self.server.gui.add_slider(
            f"Timestep", min=0, max=traj.shape[1], step=1, initial_value=0)
        @button_play.on_click
        def _(_) -> None:
            for i in range(traj.shape[1]):
                handle.position = traj[:3, i]
                handle.wxyz = np.array(
                    [traj[6, i], traj[3, i], traj[4, i], traj[5, i]])
                slider_trajectory.value = i
                time.sleep(dt)
        @button_step.on_click
        def _(_) -> None:
            if button_step.value == "Prev":
                slider_trajectory.value -= 1
            elif button_step.value == "Next":
                slider_trajectory.value += 1
        @slider_trajectory.on_update
        def _(_):
            i = slider_trajectory.value
            handle.position = traj[:3, i]
            handle.wxyz = np.array(
                [traj[6, i], traj[3, i], traj[4, i], traj[5, i]])

    def add_debug_body(self, name, position=(0,0,0),
                         radius=[1.0, 1.5, 2.0], color=(255,0,0)):
        body_mesh = trimesh.creation.icosphere(radius=0.1, subdivisions=3)
        body_mesh.apply_scale(radius)
        vertices = body_mesh.vertices
        faces = body_mesh.faces
        mesh_handle = self.server.scene.add_mesh_simple('/' + name,
                                          vertices=vertices,
                                          faces=faces,
                                          position=position,
                                          color=color)
        return mesh_handle

    def add_body_trajectory(self, label, handle, traj, dt=0.01):
        button_play = self.server.gui.add_button(f"Play {label}")
        button_step = self.server.gui.add_button_group("Frame", ("Prev", "Next"))
        slider_trajectory = self.server.gui.add_slider(
            f"Timestep", min=0, max=traj.shape[1], step=1, initial_value=0)
        @button_play.on_click
        def _(_) -> None:
            for i in range(traj.shape[1]):
                handle.position = traj[:3, i]
                handle.wxyz = np.array(
                    [traj[6, i], traj[3, i], traj[4, i], traj[5, i]])
                slider_trajectory.value = i
                time.sleep(dt)
        @button_step.on_click
        def _(_) -> None:
            if button_step.value == "Prev":
                slider_trajectory.value -= 1
            elif button_step.value == "Next":
                slider_trajectory.value += 1
        @slider_trajectory.on_update
        def _(_):
            i = slider_trajectory.value
            handle.position = traj[:3, i]
            handle.wxyz = np.array(
                [traj[6, i], traj[3, i], traj[4, i], traj[5, i]])
            
