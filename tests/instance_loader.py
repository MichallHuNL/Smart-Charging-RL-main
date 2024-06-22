import json

def load_instance(N, id=1, J=24, filename='test_instances.json'):
    """
    Load an instance from the test_instances.json file
    """
    with open(filename) as f:
      test_instances = json.load(f)


    instance_name = f"test_{N}_{id}"
    i = test_instances[instance_name]

    N, t_arr, t_dep, soc_req, soc_int, P_c_max, P_d_max, P_max_grid, E_cap, prices = i["N"], i["t_arr"], i["t_dep"], i["soc_req"], i["soc_int"], i["P_c_max"], i["P_d_max"], i["P_max_grid"], i["E_cap"], i["prices"]

    if type(soc_req) == float:
        soc_req = [soc_req] * N
    if type(soc_int) == float:
        soc_int = [soc_int] * N
    if type(P_c_max) == int:
        P_c_max = [P_c_max] * N
    if type(P_d_max) == int:
        P_d_max = [P_d_max] * N
    if type(E_cap) == int:
        E_cap = [E_cap] * N
    if type(P_max_grid) == int:
        P_max_grid = [P_max_grid] * J
    
    return N, t_arr, t_dep, soc_req, soc_int, P_c_max, P_d_max, P_max_grid, E_cap, prices