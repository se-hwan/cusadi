# TODO: Parent class with utility methods common to all isaaclab controllers

'''
Design plan:

    Init with pinocchio model and urdf
    set_parameters()
        function to set tensor with values wrt field index
    visualize()
        show single trajectory in viser for debugging
    solve_CPU()
        solve problem on CPU, for debugging
    setup_solver()
        initialize solver with solver specs
'''


# ! Need to refactor qp_backends
'''
Rough plan:
    solver.setup_parallelization()
        build_parallel_functions() -> returns and saves casadi fns, load if already built and no changes
        setup_parallel_evaluation() -> instantiates auxiliary tensors
        store parallelization settings as member vars
        return casadi_fns

    parallel_fns = *collect casadi fns from all sources
    cusadi_fns = cusadi.parallelize_functions([casadi_fns]) -> manifest should skip codegen and loading as needed

    solver.load_cusadi_fns(cusadi_fns) -> assigns member vars from keys of cusadi_fns, ignores the rest
        from stored parallelization cfg, builds cudss if needed
    
    Ready to evaluate!
'''