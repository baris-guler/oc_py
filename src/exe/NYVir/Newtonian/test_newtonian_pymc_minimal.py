import numpy as np
import pymc as pm
import pytensor.tensor as pt
from ocpy.newtonian import NewtonianModel
from ocpy.oc import Parameter

def test_minimal_pymc_node():
    print("Testing minimal PyMC node creation...")
    T0_ref = 2453174.442769
    P_ref = 0.1010159690
    
    nbody = NewtonianModel(
        central_mass = Parameter(value=0.611, fixed=True),
        T0_ref = T0_ref,
        P_ref = P_ref,
        bodies = [
            {
                "m": Parameter(value=2.283 * 0.000954588, fixed=False, std=1),
                "P": Parameter(value=3195.0, fixed=False, std=1000), 
                "e": Parameter(value=0.0, fixed=True),
                "omega": Parameter(value=0.0, fixed=True),
                "T": Parameter(value=2453619.0, fixed=False, std=10000),
            }
        ]
    )

    x = np.array([0.0, 10.0, 20.0])
    
    with pm.Model() as model:
        m_val = pm.Normal("m", mu=2.283 * 0.000954588, sigma=1e-5)
        P_val = pm.Normal("P", mu=3195.0, sigma=1.0)
        T_val = pm.Normal("T", mu=2453619.0, sigma=1.0)
        
        # This should call model_func with symbolic variables
        mu = nbody.model_func(x, b1_m=m_val, b1_P=P_val, b1_T=T_val)
        
        print("Success: mu node created without TypeError")
        print("mu type:", type(mu))
        
        # Test evaluation
        print("Evaluating mu node...")
        val = mu.eval({m_val: 2.283 * 0.000954588, P_val: 3195.0, T_val: 2453619.0})
        print("Evaluated value:", val)
        assert len(val) == len(x)

    print("Testing oc.fit() (should handle non-nuts sampler correctly)...")
    from ocpy.oc_pymc import OCPyMC
    
    oc = OCPyMC(
        oc=val,
        minimum_time_error=np.ones_like(val) * 1e-5,
        cycle=x
    )
    
    try:
        # This will call OCPyMC.fit, which uses the updated conditional kwargs
        idata = oc.fit([nbody], draws=10, tune=10, chains=1, progressbar=False)
        print("Success: oc.fit() completed without ValueError")
    except Exception as e:
        print(f"Caught error in oc.fit(): {e}")
        raise e

if __name__ == "__main__":
    try:
        test_minimal_pymc_node()
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
