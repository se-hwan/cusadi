<a id="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


<br />
<div align="center">

  <h1 align="center">CusADi</h1>

  <h3 align="center">
    Parallelizing symbolic expressions from CasADi on the GPU
    <br />
  </h3>
</div>
[![media/parallel_MPC.mp4]](media/parallel_MPC.mp4)

If you use this work, please use the following citation:

```
@article{Gaboardi2021,
    doi       = {10.21105/joss.02826},
    url       = {https://doi.org/10.21105/joss.02826},
    year      = {2021},
    publisher = {The Open Journal},
    volume    = {6},
    number    = {62},
    pages     = {2826},
    author    = {James D. Gaboardi and Sergio Rey and Stefanie Lumnitz},
    title     = {spaghetti: spatial network analysis in PySAL},
    journal   = {Journal of Open Source Software}
}
```


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about">About</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#dependencies">Dependencies</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#extensions">Extensions</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

There are many great README templates available on GitHub; however, I didn't find one that really suited my needs so I created this enhanced one. I want to create a README template so amazing that it'll be the last one you ever need -- I think this is it.

Here's why:
* Your time should be focused on creating something amazing. A project that solves a problem and helps others
* You shouldn't be doing the same tasks over and over like creating a README from scratch
* You should implement DRY principles to the rest of your life :smile:

Of course, no one template will serve all projects since your needs may be different. So I'll be adding more in the near future. You may also suggest changes by forking this repo and creating a pull request or opening an issue. Thanks to all the people have contributed to expanding this template!

Use the `BLANK_README.md` to get started.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Dependencies

`cusadi` was built on the following. Other versions may work, but are untested.

* Ubuntu 20.04
* Python 3.8+
* `casadi` 3.6.5: https://web.casadi.org/get/
* `CUDA` Toolkit: https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64
    - Version 12.3 was used, but 12.X should work
    - Add the installation to the `PATH` variable to be able to compile and find `CUDA` libraries. Make sure to replace `cuda-12.3` with the installed version:
        ```
        export PATH="/usr/local/cuda-12.3/bin:$PATH"
        export LD_LIBRARY_PATH="/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH"
        ```
* (Optional, for running benchmarks) Eigen

### Installation

1. Clone this repository (standalone, or into a larger project)
    ```
    git clone https://github.com/se-hwan/cusadi
    ```
2. (Optional) Setup a virtual environment for required Python dependencies
    ```
    python -m venv .cusadi_venv         # Python virtual environment
    source .cusadi_venv/bin/activate
    ```
3. Install `cusadi`. From the root of the cloned repository, run:
    ```
    pip install -e .
    ```
4. Compile the test function for parallelization.
    ```
    python run_codegen.py --fn=test
    ```
5. Evaluate the parallelized function for accuracy. The error should be ~1e-10 or smaller. If successful, then `cusadi` is ready for use!
    ```
    python run_cusadi_function_test.py --fn=test
    ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- USAGE EXAMPLES -->
## Usage

1. Define some symbolic `casadi` function for parallelization. This function could be the dynamics of a system, a value iteration update, a controller, etc., but do not need to be limited to optimal control applications. There are many examples and tutorials available online:
    - https://web.casadi.org/docs/
    - https://folk.ntnu.no/vladimim/#/6
    - https://www.syscop.de/files/2022ws/numopt/ex1.pdf
    - In our case, we'll parallelize the dynamics of a pendulum as a trivial example. `casadi` is available in C++, MATLAB, and Python, but we'll do this example in MATLAB
        ```
        % Add casadi to MATLAB path. Do this for wherever the casadi folder is downloaded from https://web.casadi.org/get/
        addpath(genpath('[CASADI_FOLDER_LOCATION]'));
        import casadi.*

        % Symbolic expressions
        x_pend = casadi.SX.sym('x_pend', 2, 1);             % pendulum state
        g = casadi.SX.sym('g', 1, 1);                       % pendulum parameters, gravity and length
        l = casadi.SX.sym('l', 1, 1);
        dt = casadi.SX.sym('dt', 1, 1);                     % simulation timestep

        f_pend = [x_pend(2); -g*sin(x_pend(1))/l];          % pendulum dynamics
        J_pend = jacobian(f_pend, x_pend);                  % Jacobian of pendulum dynamics w.r.t the state
        omega_next = x_pend(2) - (g*sin(x_pend(1))/l)*dt    % Semi-implicit Euler integration of dynamics
        theta_next = x_pend(1) + omega_next*dt
        x_next_pend = [theta_next; omega_next];

        % Export and save as casadi functions
        % [casadi_expr] = casadi.Function('[fn_name]', {[input1, input2, ...]}, {[output1, output2, ...]})
        fn_dynamics = casadi.Function('fn_dynamics', {x_pend, g, l}, {f_pend});
        fn_jacobian = casadi.Function('fn_jacobian', {x_pend, g, l}, {J_pend});
        fn_sim_step = casadi.Function('fn_sim_step', {x_pend, g, l, dt}, {x_next_pend});

        fn_dynamics.save('fn_dynamics.casadi')
        fn_sim_step.save('fn_sim_step.casadi')
        fn_jacobian.save('fn_jacobian.casadi')
        ```
2. Move the saved functions to `src/casadi_functions` of the `cusadi` directory.
2. Compile the functions for parallelization. From the root directory of `cusadi`:
    ```
    python run_codegen.py --fn=fn_dynamics
    python run_codegen.py --fn=fn_sim_step
    python run_codegen.py --fn=fn_jacobian
    ```

3. Evaluate the parallelized functions with `cusadi` in PyTorch
    ```
    import torch
    from cusadi import *
    from casadi import *

    BATCH_SIZE = 10000

    x0 = torch.rand((BATCH_SIZE, 2), device='cuda', dtype=torch.double)                 # Random initial states
    g = 9.81 * torch.ones((BATCH_SIZE, 1), device='cuda', dtype=torch.double)           # Gravity for each env.
    l = torch.rand((BATCH_SIZE, 1), device='cuda', dtype=torch.double)                  # Random lengths for each env.
    dt = torch.linspace(0.001, 0.1, BATCH_SIZE, device='cuda', dtype=torch.double)      # Varying timestep for each env.

    fn_casadi_sim_step = casadi.Function.load(os.path.join(CUSADI_FUNCTION_DIR, "fn_sim_step.casadi"))
    fn_cusadi_sim_step = CusadiFunction(fn_casadi_sim_step, BATCH_SIZE)
    fn_cusadi_sim_step.evaluate(x0, g, l, dt)           # Evaluate fn. with CUDA kernel 
    x_next = fn_cusadi_sim_step.outputs_sparse[0]       # Access results.
    ```
4. With this example, by putting the `evaluate()` call in a for loop, a parallel simulator can be created. You can quickly sweep the effect of the parameters (`l` and `g`) or timestep (`dt`) on the system as well.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Extensions

- [x] Release
- [ ] Support for `JAX`
- [ ] Interface with `cuBLAS`, `cuSPARSE`
- [ ] Explore CPU parallelism opportunities
- [ ] Streamline exporting, saving, and compilation flow
- [ ] IsaacGym/Orbit/IsaacLab examples
- [ ] Public examples for parallelized MPC, other optimal controllers (coming soon!)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Top contributors:

<a href="https://github.com/se-hwan/cusadi/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=se-hwan/cusadi" alt="contrib.rocks image" />
</a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Se Hwan Jeon - [@your_twitter](https://twitter.com/your_username) - email@example.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)
* [Malven's Grid Cheatsheet](https://grid.malven.co/)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)
* [React Icons](https://react-icons.github.io/react-icons/search)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/se-hwan/cusadi.svg?style=for-the-badge
[contributors-url]: https://github.com/se-hwan/cusadi/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/se-hwan/cusadi.svg?style=for-the-badge
[forks-url]: https://github.com/se-hwan/cusadi/network/members
[stars-shield]: https://img.shields.io/github/stars/se-hwan/cusadi.svg?style=for-the-badge
[stars-url]: https://github.com/se-hwan/cusadi/stargazers
[issues-shield]: https://img.shields.io/github/issues/se-hwan/cusadi.svg?style=for-the-badge
[issues-url]: https://github.com/se-hwan/cusadi/issues
[license-shield]: https://img.shields.io/github/license/se-hwan/cusadi.svg?style=for-the-badge
[license-url]: https://github.com/se-hwan/cusadi/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png