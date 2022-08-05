import numpy as np
from pymultinest import run
import scipy
import tensorflow.compat.v1 as tf
import time

emulator = tf.keras.models.load_model('emulator')

# define the meaning for the features, i.e. your model parameters
parameters = np.array([r"$\log_{10} f_{*,10}$",
                       r"$\alpha_*$",
                       r"$\log_{10} f_{\rm esc, 10}$",
                       r"$\alpha_{\rm esc}$",
                       r"$\log_{10}[M_{\rm turn}/{\rm M}_{\odot}]$",
                       r"$t_*$",
                       r"$\log_{10}\frac{L_{\rm X<2keV}/{\rm SFR}}{{\rm erg\ s^{-1}\ M_{\odot}^{-1}\ yr}}$",
                       r"$E_0/{\rm keV}$",
                       r"$\alpha_{\rm X}$"])
                       
# and their limits
limits = np.array([[-3,0], [-0.5,1], [-3,0],[-1,0.5], [8,10], [0.01,1], [38,42], [0.1,1.5], [-1,3]])

def Prior(cube, ndim, nparams):
    return cube

def log_likelihood(cube, ndim, nparams):
    theta = np.zeros(ndim)
    for i in range(ndim):
        theta[i] = cube[i]
    model = emulator.predict(theta.reshape([1,-1]))[0]
    total_sum = 0
    current_index = 0
    
    # neutral fraction
    McGreer_NF = model[current_index]
    current_index+=1
    if McGreer_NF>1.: 
        McGreer_NF=1 # physical prior
   
    McGreer_Mean = 0.06
    McGreer_OneSigma = 0.05
    if McGreer_NF>McGreer_Mean:
        total_sum += np.square((McGreer_Mean - McGreer_NF) / McGreer_OneSigma) # 1side Gaussian
        
    
    # CMB optical depth
    tau_value = model[current_index]
    current_index+=1
    
    # Mean and one sigma errors for the Planck constraints, 2006.16828
    PlanckTau_Mean = 0.0569
    PlanckTau_OneSigma_u = 0.0081
    PlanckTau_OneSigma_l = 0.0066
    total_sum += np.square( PlanckTau_Mean - tau_value )/(PlanckTau_OneSigma_u * PlanckTau_OneSigma_l +
                 (PlanckTau_OneSigma_u - PlanckTau_OneSigma_l) * (tau_value - PlanckTau_Mean))     # one way to write likelihood for 2-side Gaussian

    #z=8 and 10 21cm PS
    for redshift in [8, 10]:
        k_start = np.fromfile('HERA_Phase1_Limits/k_start_z%d.bin'%redshift, dtype=int)[0]
        ks = slice(k_start-1, None, 2)
        k_limit_vals = np.fromfile('./HERA_Phase1_Limits/PS_limit_ks_z%d.bin'%redshift)
        kwf_limit_vals = np.fromfile('./HERA_Phase1_Limits/PS_limit_kwfs_z%d.bin'%redshift)
        Nkbins = len(k_limit_vals)
        Nkwfbins = len(kwf_limit_vals)
        PS_limit_vals = np.fromfile('./HERA_Phase1_Limits/PS_limit_vals_z%d.bin'%redshift)
        PS_limit_vars = np.fromfile('./HERA_Phase1_Limits/PS_limit_vars_z%d.bin'%redshift)

        ModelPS_val = 10**model[current_index:current_index+len(PS_limit_vals)]
        current_index+=len(PS_limit_vals)

        error_val = np.sqrt(PS_limit_vars + (0.2*ModelPS_val)**2 )
        likelihood = 0.5 + 0.5 * scipy.special.erf( ( PS_limit_vals - ModelPS_val ) / (np.sqrt(2) * error_val) ) # another way to write likelihood for 1-side Gaussian
        likelihood[likelihood <= 0.0] = 1e-50
        total_sum += -2 * np.sum(np.log(likelihood))

    # UV LF, still need to interpolate to get the number density at the observed UV magnitudes
    for redshift in [6, ]:
        fLF = 'LFs/LF_obs_Bouwens_%.6f.txt'%redshift
        observation = np.loadtxt(fLF)
        observation = observation[observation[:,0]>-20]
        modelled_LF = 10**model[current_index:current_index+len(observation)]
        current_index+=len(observation)   
    
        total_sum += np.sum(np.square((observation[:,1] - modelled_LF) / observation[:,2]))
    
    return -0.5 * total_sum

t0=time.time()
sampler = run(
    log_likelihood,
    Prior,
    n_dims=len(limits),
    n_params=len(limits),
    n_live_points=400,
    resume=True,
    verbose=True,
    write_output=True,
    outputfiles_basename='./MultiNest/21CMMCEMU',
    max_iter=0,
    n_iter_before_update=10,
    importance_nested_sampling=False,
    multimodal=True,
    evidence_tolerance=0.01,
    sampling_efficiency=0.9,
    init_MPI=False,
)

te=time.time()
print('Cost %d sec'%(te-t0))
