import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import seaborn as sns
import jax
from jax import random
import jax.numpy as jnp
from datetime import date
from scipy import signal
from statsmodels.nonparametric.smoothers_lowess import lowess
import numpyro
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.diagnostics import hpdi
import numpyro.distributions as dist
import arviz as az
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from matplotlib.ticker import AutoMinorLocator
from stl import mesh
import warnings
import time
warnings.filterwarnings('ignore')
plt.rcParams.update({"text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Computer Modern Roman"]})

class EBModels:

    """
    Class that contains the mechanical model based on Euler-Bernoulli beam theory

    Parameters
    ----------
    output: string
        Structural response quantity to be returned from model. Set to 'Strain' to return strains, use 'Mises' to return von Mises stress.

    Attributes
    ----------
    model: string
        The model identifier. Set to 'M1' for the analytical model.
    
    Methods
    -------
    forward(e, dt):
        Calculated the model output

    """

    def __init__(self, output='Strain'):

        self.output = output
        self.model = 'M1'

    def forward(self, e=None, dt=None):

        """
        Calculates the selected output of the mechanical model (strain or von Mises stress)

        Parameters
        ----------
        e: jax ndarray
            Material Young's modulus
        dt: jax ndarray
            Thickness loss (uniform) of the beam specimen
        
        Returns
        -------
        out: jax ndarray
            Selected output quantity
        """

        x = 26.54
        b = 29.32
        p = -1e3
        t = 5.9

        if (self.output == 'Strain'):
            out = ((3*p*x)/(b*e*1e3*(t-dt)**2))*1e6
        else:
            out = jnp.abs(((3*p*x)/(b*e*1e3*(t-dt)**2))*e*1e3)

        return out


class SurrModels:

    """
    Class that handles training and evaluation using the polynomial regression-based surroagates of the FE models of the specimen

    Parameters
    ----------
    model: string
        Structural model identifier. Valid names are 'M2', 'M3', 'M4' and 'M5'.
    output: string
        Structural response quantity to be returned from model. Set to 'Strain' to return strains, use 'Mises' to return von Mises stress.
    vec: boolean
        When True allows for vectorized input. Default is set to False

    Attributes
    ----------
    path: string
        The home directory for any input files.
    inp_fnames: dict
        Dictionary that assigns relevant training data file names to corresponding models
    
    Methods
    -------
    DataRestructSurr(data, features):
        Method that restructures the input data to an appropriate format for further processing.
    LoadData():
        Method that loads the input data and returns them in the expected format
    GetPolyCoeffs(order):
        Method that calculates the coefficients of a polynomial of specified order (order)
    forward(e, dt):
        Calculates the specified model output
    R2():
        Calculates the r-squared metric for the trained model
    plot_resp_surface():
        Plotting function that plots the response surface from the trained model
    """

    def __init__(self, model, output='Strain', vec=False):

        self.path = r'D:\Ph.D. Thesis\Research Work\Main Project\3PB - Corrosion Experiment\Surrogate Models\\'
        self.model = model
        self.output = output
        self.vec = vec
        self.inp_fnames = {'M2': 'Inp_CCD.txt', 'M3': 'Inp_CCD_Det.txt', 'M4': 'Inp_CCD.txt', 'M5': 'Inp_CCD_Det.txt'}
        if (output == 'Strain'):
            out = 'exx'
        else:
            out = 'svm'
        self.out_fnames = {'M2': out+'_ccd_solid_int.txt', 'M3': out +'_ccd_solid_det.txt', 'M4': out +'_ccd_shell_int.txt', 'M5': out +'_ccd_shell_det.txt'}

    def DataRestructSurr(self, data, features=5):

        """
        Restructures the input data to an appropriate format for further processing.

        Parameters
        ----------
        data: ndarray
           Array containing the training targets, i.e., FE outputs
        features: float
            Number of features in the initial file. Defaults to 5.
        
        Returns
        -------
        data2d: ndarray
            Restructured data array containing training targets in a desirable ordering
        """

        cols = int(features)
        rows = int(data.shape[0]/features)

        data2d = np.zeros((rows, cols))

        for i in range(rows):

            data2d[i, :] = data[i*cols:(i+1)*cols]

        return data2d
    
    def LoadData(self):

        """
        Loads training data, calls restructuring method and outputs the input-output pairs

        Parameters
        ----------
        
        Returns
        -------
        inp: ndarray
            Array containing surrogate model training inputs
        data: ndarray
            Array containing surrogate model training outputs
        """

        inp = np.loadtxt(self.path + self.inp_fnames[self.model])
        inp = inp[1:, 1:]
        data = np.loadtxt(self.path + self.out_fnames[self.model])
        if (self.output == 'Strain'):
            data = self.DataRestructSurr(data)[:, 4]*1e6

        return inp, data
    
    def GetPolyCoeffs(self, order=2):

        """
        Calculates the polynomial coefficients of a specified order

        Parameters
        ----------
        order: float
           The order of the polynomial model. Set by default to 2.
        
        Returns
        -------
        poly_model.coef_: ndarray
            Array of polynomial coefficients.
        """        
        
        if (self.output != 'Strain'):
            order = 3
        inp, data = self.LoadData()
        poly = PolynomialFeatures(degree=order)
        in_features = poly.fit_transform(inp)
        poly_model = LinearRegression(fit_intercept=False)
        poly_model.fit(in_features, data)

        return poly_model.coef_
    
    def forward(self, e=None, dt=None):
        
        """
        Calculates the selected output of the mechanical model (strain or von Mises stress). Inherits the vec parameter from the constructor to allow for vectorized input.

        Parameters
        ----------
        e: jax ndarray
            Material Young's modulus
        dt: jax ndarray
            Thickness loss (uniform) of the beam specimen
        
        Returns
        -------
        out: jax ndarray
            Selected output quantity
        """

        coeffs = self.GetPolyCoeffs()
        coeffs = jnp.asarray(coeffs)

        if (self.output == 'Strain'):
            if self.vec is False:
                inp_feats = jnp.asarray([1, e, dt, e**2, e*dt, dt**2])
                out = jnp.matmul(coeffs, inp_feats)
            
            else:
                inp_feats = jnp.asarray([jnp.ones(dt.shape[0]), e*jnp.ones(dt.shape[0]), dt, e**2*jnp.ones(dt.shape[0]), e*jnp.ones(dt.shape[0])*dt, dt**2])
                out = jnp.matmul(coeffs, inp_feats).flatten()
            
        else:
            inp_feats = jnp.asarray([1, e, dt, e**2, e*dt, dt**2, e**3, (e**2)*dt, e*(dt**2), dt**3])
            out = jnp.matmul(coeffs, inp_feats)
        
        return out

    def R2(self):

        """
        Calculates the r-squared metric to assess the model fit

        Parameters
        ----------
        
        Returns
        -------
        r2: float
            The calculated r-squared metric
        """       

        inp, data = self.LoadData()
        data = data
        preds = jnp.zeros(data.shape[0])

        for i in range(data.shape[0]):
            preds = preds.at[i].set(self.forward(e=inp[i, 0], dt=inp[i, 1]))
        
        ssres = jnp.sum((data-preds)**2)
        sstot = jnp.sum((data - jnp.mean(data))**2)

        r2 = 1 - (ssres/sstot)

        return r2

    def plot_resp_surface(self):

        """
        This is a plotting function that generates the response surface from the trained model
        """       
        
        inp, data = self.LoadData()

        zlabel = {'Strain': r'$\epsilon_{\mathrm{xx}} \ \mathrm{[\mu \epsilon]}$', 'Mises': r'$\sigma_{\mathrm{vm}} \ \mathrm{[MPa]}$'}

        xx, yy = jnp.meshgrid(np.linspace(inp[:, 0].min(), inp[:, 0].max(), 20), np.linspace(inp[:, 1].min(), inp[:, 1].max(), 20))
        zz = jnp.zeros(xx.shape)
        
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):

                zz = zz.at[i, j].set(self.forward(e=xx[i, j], dt=yy[i, j]))
        
        ax = plt.figure(dpi=150).add_subplot(111, projection='3d')
        ax.scatter(inp[:, 0], inp[:, 1], data, cmap=plt.cm.coolwarm)
        ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, alpha=0.2, linewidth=0.5, edgecolor='b', cmap=plt.cm.coolwarm)
        ax.axis('tight')
        ax.view_init(azim=60.0, elev=30.0)
        ax.set_xlabel('$E \ \mathrm{[GPa]}$')
        ax.set_ylabel(r'$\Delta \tau \ \mathrm{[mm]}$')
        ax.set_zlabel(zlabel[self.output], rotation=90)
        plt.show()


class DataPrep:

    """
    Class that takes care of the preparation and loading of the experimental observations

    Parameters
    ----------

    Attributes
    ----------
    path: string
        The home directory for any input files.
    
    Methods
    -------
    DataSep():
        Method that separates the data according to the selected SHM task.
    """

    def __init__(self):

        self.path = r'D:\Ph.D. Thesis\Research Work\Main Project\3PB - Corrosion Experiment\Experimental Data\\'
    
    def DataSep(self, task):

        """
        Separates the data according to the selected SHM task.

        Parameters
        ----------
        task: string
            Identifier of SHM task which affects data separation. Valid names are 'BMU', 'Diagnosis' and 'Prognosis'.

        Returns
        -------
        data: DataFrame
            Pandas DataFrame containing the experimental observations according to the prescribed SHM task       
        """        

        df = pd.read_excel(self.path + 'Lab_Data.xlsx')
        df = df.drop(columns='Time')
        df.drop(df.tail(2).index, inplace = True)

        df_bmu = pd.DataFrame(data={'Time (min)': df['Time (min)'][(df['Time (min)'] > 0) & (df['Time (min)'] < 200)], 
                                'Mechanical Strain (με)': df['Mechanical Strain (με)'][(df['Time (min)'] > 0) & (df['Time (min)'] < 200)]})

        df_diag = pd.DataFrame(data={'Time (min)': df['Time (min)'][(df['Time (min)'] > 1100)], 
                                'Mechanical Strain (με)': df['Mechanical Strain (με)'][(df['Time (min)'] > 1100)]})

        df_prog = pd.DataFrame(data={'Time (min)': df['Time (min)'][(df['Time (min)'] > 200)], 
                                'Mechanical Strain (με)': df['Mechanical Strain (με)'][(df['Time (min)'] > 200)]})

        df_dict = {'BMU': df_bmu['Mechanical Strain (με)'], 'Diagnosis': df_diag['Mechanical Strain (με)'], 'Prognosis': [df_prog['Time (min)'][(df['Time (min)'] < 800)], df_prog['Mechanical Strain (με)'][(df['Time (min)'] < 800)], \
                                                                                                                           df_prog['Time (min)'][(df['Time (min)'] > 800)], df_prog['Mechanical Strain (με)'][(df['Time (min)'] > 800)]]}
        data = df_dict[task]

        return data

    def PlotTimeSeries(self):

        """
        Plotting function that returns a scatter plot of the experimental observations as a function of time alongside the rolling mean and 95% C.I.
        """       

        df = pd.read_excel(self.path + 'Lab_Data.xlsx')
        df = df.drop(columns='Time')
        df.drop(df.tail(2).index, inplace = True)
        rol_mu = df['Mechanical Strain (με)'].rolling(window=50, center=True, closed='both').mean().dropna()
        rol_sd = df['Mechanical Strain (με)'].rolling(window=50, center=True, closed='both').std().dropna()

        fig, ax = plt.subplots(dpi=100, figsize=(10, 6))

        ax.plot(df['Time (min)'].iloc[rol_mu.index], rol_mu, color='darkblue', label='Mean (Rolling)')
        ax.fill_between(x=df['Time (min)'].iloc[rol_mu.index], y1=rol_mu - 1.96*rol_sd, y2=rol_mu + 1.96*rol_sd, color='royalblue', alpha=0.3, label='95 \% C.I. (Rolling)', edgecolor=None)
        ax.scatter(df['Time (min)'].iloc[rol_mu.index], df['Mechanical Strain (με)'].iloc[rol_mu.index], color='darkmagenta', s=2., alpha=0.4, label='Strain measurements')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xlim(left=df['Time (min)'].min(), right=df['Time (min)'].max())
        ax.set_xlabel('Time (min)')
        ax.set_ylabel(r'$\varepsilon_{\mathrm{xx}} \ [\mathrm{\mu \varepsilon}]$')
        ax.legend(frameon=False)

        plt.show()

    def PlotHist(self):

        """
        Plotting function that returns histograms over the inactive phases of corrosion.
        """     
        df_p1 = self.DataSep('BMU')
        df_p2 = self.DataSep('Diagnosis')

        fig, ax = plt.subplots(1, 2, figsize=(10,5))

        sns.histplot(data=df_p1, ax=ax[0], color='tomato', alpha=0.4, label='Phase 1 - Inactive Corrosion $t \leq 200 \ \mathrm{min}$')
        ax[0].legend(frameon=False)
        ax[0].set_xlabel(r'$\varepsilon_{\mathrm{xx}} \ [\mathrm{\mu \varepsilon}]$')

        sns.histplot(data=df_p2, ax=ax[1], color='cornflowerblue', alpha=0.4, label='Phase 3 - Inactive Corrosion $t \geq 1100 \ \mathrm{min}$')
        ax[1].legend(frameon=False)
        ax[1].set_xlabel(r'$\varepsilon_{\mathrm{xx}} \ [\mathrm{\mu \varepsilon}]$')
        plt.tight_layout()
        plt.show()
    
    def PlotDetrended(self):

        df = pd.read_excel(self.path + 'Lab_Data.xlsx')
        df = df.drop(columns='Time')
        df.drop(df.tail(2).index, inplace = True)

        df_p1 = pd.DataFrame(data={'Time (min)': df['Time (min)'][(df['Time (min)'] > 0) & (df['Time (min)'] < 200)], 
                                'Mechanical Strain (με)': df['Mechanical Strain (με)'][(df['Time (min)'] > 0) & (df['Time (min)'] < 200)]})

        df_p3 = pd.DataFrame(data={'Time (min)': df['Time (min)'][(df['Time (min)'] > 1100)], 
                                'Mechanical Strain (με)': df['Mechanical Strain (με)'][(df['Time (min)'] > 1100)]})

        df_p2 = pd.DataFrame(data={'Time (min)': df['Time (min)'][(df['Time (min)'] > 200) & (df['Time (min)'] < 1100)], 
                                'Mechanical Strain (με)': df['Mechanical Strain (με)'][(df['Time (min)'] > 200) & (df['Time (min)'] < 1100)]})
        df_loess_15 = pd.DataFrame(lowess(df['Mechanical Strain (με)'], np.arange(len(df['Mechanical Strain (με)'])), frac=0.15)[:, 1], index=df['Mechanical Strain (με)'].index, columns=['value'])


        const_dtrnd_p1 = signal.detrend(df_p1['Mechanical Strain (με)'][(df['Time (min)'] > 0) & (df['Time (min)'] < 200)].values, type='constant')
        lin_dtrnd_p1 = signal.detrend(df_p1['Mechanical Strain (με)'][(df['Time (min)'] > 0) & (df['Time (min)'] < 200)].values)
        loess_dtrnd_p1 = df_p1['Mechanical Strain (με)'][(df['Time (min)'] > 0) & (df['Time (min)'] < 200)] - df_loess_15['value'][(df['Time (min)'] > 0) & (df['Time (min)'] < 200)]

        const_dtrnd_p2 = signal.detrend(df_p2['Mechanical Strain (με)'].values, type='constant')
        lin_dtrnd_p2 = signal.detrend(df_p2['Mechanical Strain (με)'].values)
        loess_dtrnd_p2 = df_p2['Mechanical Strain (με)'][(df['Time (min)'] > 200) & (df['Time (min)'] < 1100)] - df_loess_15['value'][(df['Time (min)'] > 200) & (df['Time (min)'] < 1100)]

        const_dtrnd_p3 = signal.detrend(df_p3['Mechanical Strain (με)'][(df['Time (min)'] > 1100)].values, type='constant')
        lin_dtrnd_p3 = signal.detrend(df_p3['Mechanical Strain (με)'][(df['Time (min)'] > 1100)].values)
        loess_dtrnd_p3 = df_p3['Mechanical Strain (με)'][(df['Time (min)'] > 1100)] - df_loess_15['value'][(df['Time (min)'] > 1100)]

        fig, ax = plt.subplots(3, 1, dpi=100, figsize=(10, 6))

        ax[0].plot(df['Time (min)'][(df['Time (min)'] > 0) & (df['Time (min)'] < 200)], const_dtrnd_p1, color='tomato', label='Detrended (Mean)')
        ax[0].plot(df['Time (min)'][(df['Time (min)'] > 0) & (df['Time (min)'] < 200)], lin_dtrnd_p1, color='turquoise', label='Detrended (Linear L.S)')
        ax[0].plot(df['Time (min)'][(df['Time (min)'] > 0) & (df['Time (min)'] < 200)], loess_dtrnd_p1, color='darkmagenta', label='Detrended (Polynomial L.S)')
        ax[0].xaxis.set_minor_locator(AutoMinorLocator())
        ax[0].yaxis.set_minor_locator(AutoMinorLocator())
        ax[0].set_xlabel('Time (min)')
        ax[0].set_ylabel(r'Residual strain $[\mathrm{\mu \epsilon}]$')

        ax[1].plot(df['Time (min)'][(df['Time (min)'] > 200) & (df['Time (min)'] < 1100)], const_dtrnd_p2, color='tomato', label='Detrended (Mean)')
        ax[1].plot(df['Time (min)'][(df['Time (min)'] > 200) & (df['Time (min)'] < 1100)], lin_dtrnd_p2, color='turquoise', label='Detrended (Linear L.S)')
        ax[1].plot(df['Time (min)'][(df['Time (min)'] > 200) & (df['Time (min)'] < 1100)], loess_dtrnd_p2, color='darkmagenta', label='Detrended (Polynomial L.S)')
        ax[1].xaxis.set_minor_locator(AutoMinorLocator())
        ax[1].yaxis.set_minor_locator(AutoMinorLocator())
        ax[1].set_xlabel('Time (min)')
        ax[1].set_ylabel(r'Residual strain $[\mathrm{\mu \epsilon}]$')

        ax[2].plot(df['Time (min)'][(df['Time (min)'] > 1100)], const_dtrnd_p3, color='tomato', label='Detrended (Mean)')
        ax[2].plot(df['Time (min)'][(df['Time (min)'] > 1100)], lin_dtrnd_p3, color='turquoise', label='Detrended (Linear L.S)')
        ax[2].plot(df['Time (min)'][(df['Time (min)'] > 1100)], loess_dtrnd_p3, color='darkmagenta', label='Detrended (Polynomial L.S)')
        ax[2].xaxis.set_minor_locator(AutoMinorLocator())
        ax[2].yaxis.set_minor_locator(AutoMinorLocator())
        ax[2].set_xlabel('Time (min)')
        ax[2].set_ylabel(r'Residual strain $[\mathrm{\mu \epsilon}]$')

        plt.legend(['Detrended (Mean)', 'Detrended (Linear L.S.)', 'Detrended (LOWESS Smoother)'], frameon=False, loc='lower center', bbox_to_anchor=(0.5, 3.5), ncol=3)

        plt.show()     

class BMU(DataPrep, EBModels, SurrModels):

    """
    Class for Bayesian Model Updating (BMU), which in this case consists of estimating the Young's modulus of the specimen material using strain response data
    obtained from the structure in its intact state.

    Parameters
    ----------
    model: string
        Identifier of the structural model used to simulate the structural response. Valid names are 'M1', 'M2', 'M3', 'M4' and 'M5'.

    Attributes
    ----------
    fwd_model:
        The structural model class used to map Young's modulus realizations to strain response values.
    data: ndarray
        The array containing experimental observations
    
    Methods
    -------
    model():
        The Bayesian model employed for BMU built using Numpyro. Specifies the Bayesian priors and likelihood for the problem.
    run_mcmc(num_samples):
        Implements the NUTS algorithm to perform Bayesian inference. 
    get_posterior(mcmc):
        Returns samples from the posterior distribution using the inference object.
    get_summary(mcmc):
        Returns basic summary statistics from the posterior distribution obtained using the inference object.
    plot_full(mcmc):
        Plotting function for KDE-based posterior densities and trace plots.
    get_maps(mcmc):
        Returns the MAP estimates from the posterior samples.
    nmse(e):
        Returns the NMSE of the strain response posterior, calculated through the Young's modulus posterior and the structural model, 
        compared to the observations.
    exp_util_nmse(mcmc, num_samples):
        Returns the NMSE-based expected utility of the decision to employ a particular structural model for the BMU task.
    logl(e, sigma):
        Returns the log-likelihood of the strain response posterior, calculated through the Young's modulus posterior and the structural model, 
        compared to the observations and accounting for the prediction error estimate.
    exp_util_logl(mcmc, num_samples):
        Returns the likelihood-based expected utility of the decision to employ a particular structural model for the BMU task.
    """

    def __init__(self, model):

        if (model == 'M1'):
            self.fwd_model = EBModels()
        else:
            self.fwd_model = SurrModels(model)
        
        self.data = DataPrep().DataSep('BMU').values

    def model(self):

        """
        Specifies the Bayesian model structure used for the BMU task.

        Parameters
        ----------
        
        Returns
        -------    
        """ 

        e = numpyro.sample('$E$', dist.Uniform(170.0, 238.05))
        sigma = numpyro.sample('$\sigma$', dist.Uniform(0.01, 10))
        mu = self.fwd_model.forward(e=e, dt=0)

        numpyro.sample('obs', dist.Normal(mu, sigma), obs=self.data)

    def run_mcmc(self, num_samples=2000):

        """
        Implements the NUTS algorithm to perform Bayesian inference.

        Parameters
        ----------
        num_samples: int
            Number of posterior samples. Defaults to 2000.
        
        Returns
        -------
        mcmc: Numpyro object
            The inference object that contains the results of the NUTS implementation.       
        """ 

        nuts_kernel = NUTS(self.model, target_accept_prob=0.90)
        mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_chains=8, num_warmup=2000)
        rng_key = random.PRNGKey(0)
        mcmc.run(rng_key)

        return mcmc

    def get_posterior(self, mcmc):

        """
        Returns the posterior distribution sample.

        Parameters
        ----------
        mcmc: Numpyro object
            The inference object that contains the results of the NUTS implementation.
        
        Returns
        -------
        post_samp: dict
            Dictionary containing the posterior samples for each parameter.       
        """ 

        post_samp = mcmc.get_samples()

        return post_samp

    def get_summary(self, mcmc):

        """
        Returns summary statistics from the posterior distribution sample.

        Parameters
        ----------
        mcmc: Numpyro object
            The inference object that contains the results of the NUTS implementation.
        
        Returns
        -------
        summary: DataFrame
            Pandas DataFrame containing summary statistics from the posterior sample.       
        """         

        data = az.from_numpyro(mcmc)
        summary = az.summary(data)

        return summary
    
    def plot_full(self, mcmc):

        """
        Plotting function for KDE-based posterior densities and trace plots.

        Parameters
        ----------
        mcmc: Numpyro object
            The inference object that contains the results of the NUTS implementation.       
        """         

        data = az.from_numpyro(mcmc)
            
        az.plot_trace(data, compact=False, divergences=None)
        plt.tight_layout()
        plt.show()
    
    def get_maps(self, mcmc):

        """
        Function that returns the MAP estimates from the posterior distribution sample.

        Parameters
        ----------
        mcmc: Numpyro object
            The inference object that contains the results of the NUTS implementation.       
        
        Returns
        -------
        map: dict
            A dictionary containing the MAP estimate from the posterior sample of each parameter.
        """                 

        trace = az.from_numpyro(mcmc)

        grid_e, pdf_e = az.kde(trace.posterior['$E$'].values.reshape(-1,1).flatten())
        map_e = grid_e[np.argmax(pdf_e)]

        grid_sd, pdf_sd = az.kde(trace.posterior['$\sigma$'].values.reshape(-1,1).flatten())
        map_sd = grid_sd[np.argmax(pdf_sd)]

        map = {'$E$': map_e, '$\sigma$': map_sd}

        return map
    
    def nmse(self, e):

        """
        Returns the NMSE of the strain response posterior, calculated through the Young's modulus posterior and the structural model, 
        compared to the observations.

        Parameters
        ----------
        e: jax ndarray
            Posterior sample for the Young's modulus.      
        
        Returns
        -------
        nmse: float
            NMSE of the strain response posterior.
        """ 

        nmse = (100/(self.data.var()*len(self.data)))*((self.fwd_model.forward(e=e, dt=0)-self.data).T@(self.fwd_model.forward(e, dt=0)-self.data))

        return nmse

    def exp_util_nmse(self, mcmc, num_samples=2000):

        """
        Returns the NMSE-based expected utility of the decision to employ a particular structural model for the BMU task.

        Parameters
        ----------
        mcmc: Numpyro object
            The inference object that contains the results of the NUTS implementation.
        num_samples: int
            Number of samples from the posterior across different chains to use to calculate the expected utility
        
        Returns
        -------
        exp_util: float
            NMSE-based expected utility
        """ 
        
        bmu_nmse = self.nmse
        posterior = mcmc.get_samples()
        util = 0

        for i in range(posterior['$E$'][:num_samples].shape[0]):

            nmse = bmu_nmse(posterior['$E$'][i])
            util += 1 -(1/100)*nmse
        exp_util = util.item()/posterior['$E$'][:num_samples].shape[0]

        return exp_util
    
    def logl(self, e, sigma):

        """
        Returns the log-likelihood of the strain response posterior, calculated through the Young's modulus posterior and the structural model, 
        compared to the observations and accounting for the prediction error estimate.

        Parameters
        ----------
        e: jax ndarray
            Posterior sample for the Young's modulus.  
        sigma: jax ndarray
            Posterior sample for the prediction error standard deviation.  
        
        Returns
        -------
        logl: float
            Log-likelihood of the strain response posterior.
        """ 
        
        logl = jax.scipy.stats.multivariate_normal.logpdf(x=self.data - self.fwd_model.forward(e=e, dt=0), mean=jnp.zeros(len(self.data)), cov=jnp.identity(len(self.data))*sigma)

        return logl
    
    def exp_util_logl(self, mcmc, num_samples=2000):

        """
        Returns the likelihood-based expected utility of the decision to employ a particular structural model for the BMU task.

        Parameters
        ----------
        mcmc: Numpyro object
            The inference object that contains the results of the NUTS implementation.
        num_samples: int
            Number of samples from the posterior across different chains to use to calculate the expected utility.
        
        Returns
        -------
        exp_util: float
            Likelihood-based expected utility
        """

        bmu_logl = self.logl
        posterior = mcmc.get_samples()

        grid_e, pdf_e = az.kde(posterior['$E$'][:num_samples])
        map_e = grid_e[np.argmax(pdf_e)]

        grid_sd, pdf_sd = az.kde(posterior['$\sigma$'][:num_samples])
        maps_sd = grid_sd[np.argmax(pdf_sd)]

        logl_map = bmu_logl(map_e, maps_sd)

        util = 0

        for i in range(posterior['$E$'][:num_samples].shape[0]):

            logl = bmu_logl(posterior['$E$'][i], posterior['$\sigma$'][i])
            xval = jnp.abs((logl - logl_map))/jnp.maximum(jnp.abs(logl_map), jnp.abs(logl_map))
            util += 1 - xval
        
        exp_util = util.item()/posterior['$E$'][:num_samples].shape[0]

        return exp_util


class DamageDiag(DataPrep, EBModels, SurrModels):

    """
    Class for damage diagnosis, which in this case consists of estimating the thickness loss of the specimen using strain response data obtained from the 
    structure in its damaged state. The thickness loss itself depends on the model parameterization.

    Parameters
    ----------
    model: string
        Identifier of the structural model used to simulate the structural response. Valid names are 'M1', 'M2', 'M3', 'M4' and 'M5'.

    Attributes
    ----------
    fwd_model:
        The structural model class used to map Young's modulus realizations to strain response values.
    svm_model:
        The structural model class used to map thickness loss realizations to von Mises stress values.
    up_lim: float
        The upper limit of the prior range of the thickness loss parameter.
    ym: float
        The Young's modulus MAP estimate obtained for the corresponding model class from the BMU task.
    data: ndarray
        The array containing experimental observations
    
    Methods
    -------
    model():
        The Bayesian model employed for damage diagnosis built using Numpyro. Specifies the Bayesian priors and likelihood for the problem.
    run_mcmc(num_samples):
        Implements the NUTS algorithm to perform Bayesian inference. 
    get_posterior(mcmc):
        Returns samples from the posterior distribution using the inference object.
    get_summary(mcmc):
        Returns basic summary statistics from the posterior distribution obtained using the inference object.
    plot_full(mcmc):
        Plotting function for KDE-based posterior densities and trace plots.
    get_maps(mcmc):
        Returns the MAP estimates from the posterior samples.
    pf_samp(svm):
        Returns the probability of failure for a particular maximum von Mises stress value and a Gaussian yield stress distribution.
    nmse(dt):
        Returns the NMSE of the strain response posterior, calculated through the thickness loss posterior and the structural model, 
        compared to the observations.
    exp_util_nmse(mcmc, num_samples):
        Returns the NMSE-based expected utility of the decision to employ a particular structural model for the damage diagnosis task, when assessed in terms of 
        output reconstruction.
    logl(dt, sigma):
        Returns the log-likelihood of the strain response posterior, calculated through the thickness loss posterior and the structural model, 
        compared to the observations and accounting for the prediction error estimate.
    exp_util_logl(mcmc, num_samples):
        Returns the likelihood-based expected utility of the decision to employ a particular structural model for the damage diagnosis task, when assessed in terms
        of output reconstruction.
    exp_util_pf(mcmc, num_samples):
        Returns the expected utility of the decision to employ a particular structural model for the damage diagnosis task, when assessed in terms of the structural 
        reliability estimate compared to an oracle model.
    comb_util(mcmc, w1, w2):
        Weighted expected utility of the decision to employ a particular structural model for the damage diagnosis task.
    plot_comb_util():
        Plotting function that plots the combined expected utility as a function of the weighting coefficients.
    """

    def __init__(self, model):
                
        if (model == 'M1'):
            self.fwd_model = EBModels()
            self.svm_model = EBModels(output='Mises')
        else:
            self.fwd_model = SurrModels(model)
            self.svm_model = SurrModels(model, output='Mises')

        upper_lims = {'M1': 1.0, 'M2': 1.0, 'M3': 1.4, 'M4': 1.0, 'M5': 1.4}
        es = {'M1': 188.14, 'M2': 173.64, 'M3': 173.38, 'M4': 176.34, 'M5': 176.30}
        self.up_lim = upper_lims[model]
        self.ym = es[model]
        self.data = DataPrep().DataSep('Diagnosis').values

    def model(self):

        """
        Specifies the Bayesian model structure used for the damage diagnosis task.

        Parameters
        ----------
        
        Returns
        -------    
        """

        t = numpyro.sample(r'$\Delta \tau$', dist.Uniform(0.0, self.up_lim))
        sigma = numpyro.sample('$\sigma$', dist.Uniform(0.01, 10))
        mu = self.fwd_model.forward(e=self.ym, dt=t)

        numpyro.sample('obs', dist.Normal(mu, sigma), obs=self.data)

    def run_mcmc(self, num_samples=2000):

        """
        Implements the NUTS algorithm to perform Bayesian inference.

        Parameters
        ----------
        num_samples: int
            Number of posterior samples. Defaults to 2000.
        
        Returns
        -------
        mcmc: Numpyro object
            The inference object that contains the results of the NUTS implementation.       
        """

        nuts_kernel = NUTS(self.model, target_accept_prob=0.90)
        mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_chains=8, num_warmup=2000)
        rng_key = random.PRNGKey(0)
        mcmc.run(rng_key)

        return mcmc

    def get_posterior(self, mcmc):

        """
        Returns the posterior distribution sample.

        Parameters
        ----------
        mcmc: Numpyro object
            The inference object that contains the results of the NUTS implementation.
        
        Returns
        -------
        post_samp: dict
            Dictionary containing the posterior samples for each parameter.       
        """ 

        post_samp = mcmc.get_samples()

        return post_samp

    def get_summary(self, mcmc):

        """
        Returns summary statistics from the posterior distribution sample.

        Parameters
        ----------
        mcmc: Numpyro object
            The inference object that contains the results of the NUTS implementation.
        
        Returns
        -------
        summary: DataFrame
            Pandas DataFrame containing summary statistics from the posterior sample.       
        """       

        data = az.from_numpyro(mcmc)

        return az.summary(data)
    
    def plot_full(self, mcmc):

        """
        Plotting function for KDE-based posterior densities and trace plots.

        Parameters
        ----------
        mcmc: Numpyro object
            The inference object that contains the results of the NUTS implementation.       
        """         

        data = az.from_numpyro(mcmc)
            
        az.plot_trace(data, compact=False, divergences=None)
        plt.tight_layout()
        plt.show()
    
    def get_maps(self, mcmc):

        """
        Function that returns the MAP estimates from the posterior distribution sample.

        Parameters
        ----------
        mcmc: Numpyro object
            The inference object that contains the results of the NUTS implementation.       
        
        Returns
        -------
        map: dict
            A dictionary containing the MAP estimate from the posterior sample of each parameter.
        """ 

        trace = az.from_numpyro(mcmc)

        grid_t, pdf_t = az.kde(trace.posterior[r'$\Delta \tau$'].values.reshape(-1,1).flatten())
        map_t = grid_t[np.argmax(pdf_t)]

        grid_sd, pdf_sd = az.kde(trace.posterior['$\sigma$'].values.reshape(-1,1).flatten())
        map_sd = grid_sd[np.argmax(pdf_sd)]

        map = {r'$\Delta \tau$': map_t, '$\sigma$': map_sd}

        return map
    
    def pf_samp(self, svm):

        """
        Returns the probability of failure for a particular maximum von Mises stress value and a Gaussian yield stress distribution.

        Parameters
        ----------
        svm: float
            Maximum von Mises stress.   
        
        Returns
        -------
        pf_samp: float
            The probability of failure under a Gaussian distribution of the yield stress.
        """ 

        pf = jax.scipy.stats.norm.cdf(x=svm, loc=284.5, scale=21.5)

        return pf
    
    def nmse(self, dt):

        """
        Returns the NMSE of the strain response posterior, calculated through the thickness loss posterior and the structural model, 
        compared to the observations.

        Parameters
        ----------
        dt: jax ndarray
            Posterior sample for the thickness loss of the specimen. 
        
        Returns
        -------
        nmse: float
            NMSE of the strain response posterior.
        """ 

        nmse = (100/(self.data.var()*len(self.data)))*((self.fwd_model.forward(e=self.ym, dt=dt)-self.data).T@(self.fwd_model.forward(e=self.ym, dt=dt)-self.data))

        return nmse

    def exp_util_nmse(self, mcmc, num_samples=2000):

        """
        Returns the NMSE-based expected utility of the decision to employ a particular structural model for the damage diagnosis task, when assessed in terms of 
        output reconstruction.

        Parameters
        ----------
        mcmc: Numpyro object
            The inference object that contains the results of the NUTS implementation.
        num_samples: int
            Number of samples from the posterior across different chains to use to calculate the expected utility
        
        Returns
        -------
        exp_util: float
            NMSE-based expected utility
        """ 

        posterior = mcmc.get_samples()
        diag_nmse = self.nmse
        util = 0

        for i in range(len(posterior[r'$\Delta \tau$'][:num_samples])):

            nmse = diag_nmse(posterior[r'$\Delta \tau$'][i])
            util += 1 -(1/100)*nmse

        exp_util = util/len(posterior[r'$\Delta \tau$'][:num_samples])

        return exp_util
    
    def logl(self, dt, sigma):
        
        """
        Returns the log-likelihood of the strain response posterior, calculated through the thickness loss posterior and the structural model, 
        compared to the observations and accounting for the prediction error estimate.

        Parameters
        ----------
        dt: jax ndarray
            Posterior sample for the thickness loss of the specimen.  
        sigma: jax ndarray
            Posterior sample for the prediction error standard deviation.  
        
        Returns
        -------
        logl: float
            Log-likelihood of the strain response posterior.
        """ 

        logl = jax.scipy.stats.multivariate_normal.logpdf(x=self.data -self.fwd_model.forward(e=self.ym, dt=dt), mean=np.zeros(len(self.data)), cov=np.identity(len(self.data))*sigma)

        return logl

    def exp_util_logl(self, mcmc, num_samples=2000):

        """
        Returns the likelihood-based expected utility of the decision to employ a particular structural model for the damage diagnosis task, when assessed in terms
        of output reconstruction.

        Parameters
        ----------
        mcmc: Numpyro object
            The inference object that contains the results of the NUTS implementation.
        num_samples: int
            Number of samples from the posterior across different chains to use to calculate the expected utility.
        
        Returns
        -------
        exp_util: float
            Likelihood-based expected utility
        """

        diag_logl = self.logl
        posterior = mcmc.get_samples()

        grid_dt, pdf_dt = az.kde(posterior[r'$\Delta \tau$'][:num_samples])
        map_dt = grid_dt[np.argmax(pdf_dt)]

        grid_sd, pdf_sd = az.kde(posterior['$\sigma$'][:num_samples])
        maps_sd = grid_sd[np.argmax(pdf_sd)]

        logl_map = diag_logl(map_dt, maps_sd)
        util = 0

        for i in range(posterior[r'$\Delta \tau$'][:num_samples].shape[0]):

            logl = diag_logl(posterior[r'$\Delta \tau$'][i], posterior['$\sigma$'][i])
            xval = np.abs((logl - logl_map)/logl_map)
            util += 1 - xval

        exp_util = util/(posterior[r'$\Delta \tau$'][:num_samples].shape[0])

        return exp_util
    
    def exp_util_pf(self, mcmc, num_samples=2000):

        """
        Returns the expected utility of the decision to employ a particular structural model for the damage diagnosis task, when assessed in terms of the structural 
        reliability estimate compared to an oracle model.

        Parameters
        ----------
        mcmc: Numpyro object
            The inference object that contains the results of the NUTS implementation.
        num_samples: int
            Number of samples from the posterior across different chains to use to calculate the expected utility.
        
        Returns
        -------
        exp_util: float
            Reliability-based expected utility.
        """

        posterior = mcmc.get_samples()
        post_orc = DamageDiag('M3').run_mcmc().get_samples()
        svm_post = jnp.zeros(posterior[r'$\Delta \tau$'][:num_samples].shape[0])
        svm_orc = jnp.zeros(post_orc[r'$\Delta \tau$'][:num_samples].shape[0])

        for i in range(posterior[r'$\Delta \tau$'][:num_samples].shape[0]):

            svm_post = svm_post.at[i].set(self.svm_model.forward(e=self.ym, dt=posterior[r'$\Delta \tau$'][i]))
            svm_orc = svm_orc.at[i].set(SurrModels('M3', 'Mises').forward(e=self.ym, dt=post_orc[r'$\Delta \tau$'][i]))

        pf_pred = np.log10(self.pf_samp(svm_post))
        pf_orc = np.log10(self.pf_samp(svm_orc))
        pf_max = np.maximum(np.abs(pf_pred), np.abs(pf_orc))

        exp_util = (1 - np.abs(pf_orc - pf_pred)/pf_max).mean()

        return exp_util

    def comb_util(self, mcmc, w1=0.5, w2=0.5):

        """
        Weighted expected utility of the decision to employ a particular structural model for the damage diagnosis task.

        Parameters
        ----------
        mcmc: Numpyro object
            The inference object that contains the results of the NUTS implementation.
        w1: float
            Relative weight of the data-based expected utility.
        w2: float
            Relative weight of the reliability-based expected utility.
        
        Returns
        -------
        comb_util: float
            Combined expected utility
        """

        util = w1*self.exp_util_logl(mcmc) + w2*self.exp_util_pf(mcmc)

        return util

    def plot_comb_util(self):

        """
        Plotting function that plots the combined expected utility as a function of the weighting coefficients.

        Parameters
        ----------
        
        Returns
        -------
        """

        mcmc = self.run_mcmc()
        w = jnp.array([jnp.linspace(0, 1, 10), 1 - jnp.linspace(0, 1, 10)])
        w = jnp.concatenate([w, jnp.flip(w, axis=0)], axis=1)
        comb_util_surf = jnp.zeros(w.shape[1])

        logl_util = self.exp_util_logl(mcmc)
        pf_util = self.exp_util_pf(mcmc)

        for i in range(w.shape[1]):

                comb_util_surf = comb_util_surf.at[i].set(w[0, i]*logl_util + w[1, i]*pf_util)
        
        ax = plt.figure(dpi=150).add_subplot(111, projection='3d')
        ax.scatter(w[0, :], w[1, :], comb_util_surf, cmap=plt.cm.coolwarm)
        ax.axis('tight')
        ax.view_init(azim=60.0, elev=30.0)
        ax.set_xlabel('$w_1$')
        ax.set_ylabel('$w_2$')
        ax.set_zlabel('Combined Utility', rotation=90)
        ax.set_zlim(top=1.05)
        plt.show()


class StructDetMon(DataPrep, EBModels, SurrModels):

    """
    Class for structural deterioration monitoring, which in this case consists of estimating the parameters of a deterioration model describing the temporal 
    evolution of thickness loss on the specimen using strain response data. The thickness loss itself depends on the model parameterization.

    Parameters
    ----------
    model: string
        Identifier of the structural model used to simulate the structural response. Valid names are 'M1', 'M2', 'M3', 'M4' and 'M5'.
    det_model: string
        Identifier of the deterioration model describing the thickness loss evolution. Valid names are 'Logistic' and 'Exponential'.

    Attributes
    ----------
    fwd_model:
        The structural model class used to map Young's modulus realizations to strain response values.
    svm_model:
        The structural model class used to map thickness loss realizations to von Mises stress values.
    det_mod_id: string
        The deterioration model identifier. Inherits from the model parameter of the class.
    det_model:
        The deterioration model describing the temporal evolution. Either a logistic-type or exponential-type model depending on the chosen class parameter.
    mod_id:
        Structural model identifier. Inherits from the model parameter of the class.
    up_lim: float
        The upper limit of the prior range of the thickness loss parameter.
    ym: float
        The Young's modulus MAP estimate obtained for the corresponding model class from the BMU task.
    data: DataFrame
        A DataFrame containing experimental observations.
    tr_times: ndarray
        Array containinig the time instances for the training set.
    tr_obs: ndarray
        Array containing the strain response measurements for the training set.
    tst_times: ndarray
        Array containinig the time instances for the test set.
    tst_obs: ndarray
        Array containing the strain response measurements for the test set.
    
    Methods
    -------
    log_det_model(t, a, b, c):
        A three parameter logistic-type deterioration model describing thickness loss over time.
    exp_det_model(t, a, b):
        A two parameter exponential-type deterioration model describing thickness loss over time.
    model(times, strains):
        The Bayesian model employed for damage diagnosis built using Numpyro. Specifies the Bayesian priors and likelihood for the problem.
    run_mcmc(num_samples, oracle):
        Implements the NUTS algorithm to perform Bayesian inference. 
    get_posterior(mcmc):
        Returns samples from the posterior distribution using the inference object.
    get_summary(mcmc):
        Returns basic summary statistics from the posterior distribution obtained using the inference object.
    post_predicitve(mcmc, num_samples):
        Returns a sample from the posterior predictive distribution over the entire timeseries.
    plot_full(mcmc):
        Plotting function for KDE-based posterior densities and trace plots.
    post_summary(mcmc, num_samples):
        Returns a summary of characteristic statistical quantities from the posterior and predictive processes.
    pf_samp(svm):
        Returns the probability of failure for a particular maximum von Mises stress value and a Gaussian yield stress distribution.
    nmse(str_post, obs):
        Returns the NMSE of the posterior predictive process (strain response), obtained through the posterior thickness loss process and the structural model, 
        compared to the observations.
    exp_util_nmse(mcmc, num_samples, set):
        Returns the NMSE-based expected utility of the decision to employ a particular structural model for the structural deterioration monitoring task, 
        when assessed in terms of the predictive process.
    logl(str_post, sig_post, obs):
        Returns the log-likelihood of the posterior predictive process (strain response), obtained through the posterior thickness loss process and the structural model, 
        compared to the observations and accounting for the prediction error estimate.
    exp_util_logl(mcmc, num_samples):
        Returns the likelihood-based expected utility of the decision to employ a particular structural model for the structural deterioration monitoring task, 
        when assessed in terms of the predictive process.
    exp_util_pf(mcmc, num_samples, set):
        Returns the expected utility of the decision to employ a particular structural model for the structural deterioration monitoring task, when assessed in terms of the 
        structural reliability estimate at a specific time instance compared to an oracle model.
    comb_util(mcmc, set, w1, w2):
        Weighted expected utility of the decision to employ a particular structural model for the structural deterioration monitoring task.
    plot_post_det_process(mcmc):
        Plotting function that plots the posterior deterioration (thickness loss) process.
    plot_tr_str_post(mcmc):
        Plotting function that plots the predictive process (strain response) over the training set.
    plot_post_pred(mcmc, num_samples)
        Plotting function that plots the posterior predictive process (strain response) over the entire timeseries.
    """

    def __init__(self, model, det_model='Logistic'):

        if (model == 'M1'):
            self.fwd_model = EBModels()
            self.svm_model = EBModels(output='Mises')
        else:
            self.fwd_model = SurrModels(model, vec=True)
            self.svm_model = SurrModels(model, output='Mises', vec=True)
        
        det_models = {'Logistic': self.log_det_model, 'Exponential': self.exp_det_model}
        self.det_mod_id = det_model
        self.det_model = det_models[det_model]
        es = {'M1': 188.14, 'M2': 173.64, 'M3': 173.38, 'M4': 176.34, 'M5': 176.30}
        upper_lims = {'M1': 1.0, 'M2': 1.0, 'M3': 1.4, 'M4': 1.0, 'M5': 1.4}
        self.mod_id = model
        self.up_lim = upper_lims[model]
        self.ym = es[model]
        self.data = DataPrep().DataSep('Prognosis')
        self.tr_times = self.data[0].values
        self.tr_obs = self.data[1].values
        self.tst_times = self.data[2].values
        self.tst_obs = self.data[3].values

    def log_det_model(self, t, a, b, c):

        """
        Defines a three parameter logistic-type deterioration model.

        Parameters
        ----------
        t: ndarray
            Array containing time instances
        a: float
            Deterioration model parameter
        b: float
            Deterioration model parameter
        c: float
            Deterioration model parameter

        Returns
        -------
        out: jax ndarray
            Array containing thickness loss over time according to the deterioration model.    
        """

        out = (c/(1+jnp.exp(-(b+a*t))))

        return out

    def exp_det_model(self, t, a, b):

        """
        Defines a three parameter logistic-type deterioration model.

        Parameters
        ----------
        t: ndarray
            Array containing time instances
        a: float
            Deterioration model parameter
        b: float
            Deterioration model parameter

        Returns
        -------
        out: jax ndarray
            Array containing thickness loss over time according to the deterioration model.    
        """

        tmin = 200
        tmax = 800
        a += 1e-4

        out = a*(1/(tmax-tmin))*(t-200)**b

        return out

    def model(self, times=None, strains=None):

        """
        Specifies the Bayesian model structure used for the structural deterioration monitoring task.

        Parameters
        ----------
        times: ndarray
            Array of time instances.
        strains: ndarray
            Array of strain observations.

        Returns
        -------    

        """

        if (self.det_mod_id == 'Logistic'):
            a = numpyro.sample(r'$\alpha$', dist.HalfNormal(0.1))
            b = numpyro.sample(r'$\beta$', dist.Normal(0.0, 1.0))
            c = numpyro.sample(r'$\gamma$', dist.Uniform(0.01, self.up_lim))
            dt = self.det_model(times, a, b, c)

        else:
            a = numpyro.sample(r'$\alpha$', dist.Uniform(0.1, 1.5))
            b = numpyro.sample(r'$\beta$', dist.Uniform(-1.5, 1.5))
            dt = self.det_model(times, a, b)

        sigma = numpyro.sample('$\sigma$', dist.Uniform(0.01, 10.0))
        mu = self.fwd_model.forward(e=self.ym, dt=dt).flatten()

        numpyro.sample('obs', dist.Normal(mu, sigma), obs=strains)
    
    def run_mcmc(self, num_samples=2000, oracle=False):

        """
        Implements the NUTS algorithm to perform Bayesian inference.

        Parameters
        ----------
        num_samples: int
            Number of posterior samples. Defaults to 2000.
        oracle: Boolean
            Boolean variable set to False. When True it specifies that the structural model is the oracle, which differentiates the training set.

        Returns
        -------    
        mcmc: Numpyro object
            The inference object that contains the results of the NUTS implementation.
        """

        if oracle:
            times = jnp.concatenate([self.tr_times, self.tst_times], axis=0)
            obs = jnp.concatenate([self.tr_obs, self.tst_obs], axis=0)

        else:
            times = self.tr_times
            obs = self.tr_obs

        nuts_kernel = NUTS(self.model, target_accept_prob=0.90)
        mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_chains=8, num_warmup=2000)
        rng_key = random.PRNGKey(0)
        mcmc.run(rng_key, times=times, strains=obs)

        return mcmc

    def get_posterior(self, mcmc):

        """
        Returns the posterior distribution sample.

        Parameters
        ----------
        mcmc: Numpyro object
            The inference object that contains the results of the NUTS implementation.
        
        Returns
        -------
        post_samp: dict
            Dictionary containing the posterior samples for each parameter.       
        """ 

        post_samp = mcmc.get_samples()

        return post_samp

    def get_summary(self, mcmc):

        """
        Returns summary statistics from the posterior distribution sample.

        Parameters
        ----------
        mcmc: Numpyro object
            The inference object that contains the results of the NUTS implementation.
        
        Returns
        -------
        summary: DataFrame
            Pandas DataFrame containing summary statistics from the posterior sample.       
        """       

        data = az.from_numpyro(mcmc)

        return az.summary(data)
    
    def post_predicitve(self, mcmc, num_samples=1000):

        """
        Returns a sample from the posterior predictive distribution over the entire timeseries.

        Parameters
        ----------
        mcmc: Numpyro object
            The inference object that contains the results of the NUTS implementation.
        num_samples: int
            Number of samples to draw from the posterior predictive process.
        
        Returns
        -------
        post_preds: jax ndarray
            Array containing samples from the posterior predictive process.    
        """       

        times = jnp.concatenate([self.tr_times, self.tst_times], axis=0)

        rng_key = random.PRNGKey(0)
        post_pred = Predictive(self.model, self.get_posterior(mcmc), num_samples=num_samples)
        post_preds = post_pred(rng_key=rng_key, times=times)['obs']

        return post_preds
    
    def plot_full(self, mcmc):

        """
        Plotting function for KDE-based posterior densities and trace plots.

        Parameters
        ----------
        mcmc: Numpyro object
            The inference object that contains the results of the NUTS implementation.       
        """         

        data = az.from_numpyro(mcmc)
        
        az.plot_trace(data, compact=False, divergences=None)
        plt.tight_layout()
        plt.show()
    
    def post_summary(self, mcmc, num_samples=2000):

        """
        Returns a summary of characteristic statistical quantities from the posterior and predictive processes.

        Parameters
        ----------
        mcmc: Numpyro object
            The inference object that contains the results of the NUTS implementation.
        num_samples: int
            Number of samples from the posterior across different chains.

        Returns
        -------
        post_dt_mean: jax ndarray
            Mean of the posterior deterioration process.
        post_dt_hpdi: tuple
            Tuple of arrays containing the 95% high probability density interval of the posterior deterioration process.
        post_str_mean: jax ndarray
            Mean of the predicted posterior process. 
        post_str_hpdi: jax ndarray               
            Tuple of arrays containing the 95% high probability density interval of the predicted posterior process.
        post_sig_mean: float
            Posterior mean of the prediction error standard deviation.
        """         

        posterior = mcmc.get_samples()

        if (self.det_mod_id == 'Logistic'):
            post_dt = self.det_model(self.tr_times, jnp.expand_dims(posterior[r'$\alpha$'], -1)[:num_samples], jnp.expand_dims(posterior[r'$\beta$'], -1)[:num_samples], \
                                     jnp.expand_dims(posterior[r'$\gamma$'], -1)[:num_samples])
            
        else:
            post_dt = self.det_model(self.tr_times, jnp.expand_dims(posterior[r'$\alpha$'], -1)[:num_samples], jnp.expand_dims(posterior[r'$\beta$'], -1)[:num_samples])            

        post_dt_mean = jnp.mean(post_dt, axis=0)
        post_dt_hpdi = hpdi(post_dt, 0.95)
        post_sig = jnp.expand_dims(posterior[r'$\sigma$'], -1)
        post_sig_mean = jnp.mean(post_sig, axis=0)

        post_str = jnp.zeros(post_dt.shape)

        for i in range(post_str.shape[0]):
            post_str = post_str.at[i].set(self.fwd_model.forward(e=self.ym, dt=post_dt[i, :]))

        post_str_mean = jnp.mean(post_str, axis=0)
        post_str_hpdi = hpdi(post_str, 0.95)

        return post_dt_mean, post_dt_hpdi, post_str_mean, post_str_hpdi, post_sig_mean
    
    def pf_samp(self, svm):

        """
        Returns the probability of failure for a particular maximum von Mises stress value and a Gaussian yield stress distribution.

        Parameters
        ----------
        svm: float
            Maximum von Mises stress.   
        
        Returns
        -------
        pf_samp: float
            The probability of failure under a Gaussian distribution of the yield stress.
        """ 

        pf = jax.scipy.stats.norm.cdf(x=svm, loc=284.5, scale=21.5)

        return pf
    
    def nmse(self, str_post, obs):

        """
        Returns the NMSE of the posterior predictive process (strain response), obtained through the posterior thickness loss process and the structural model, 
        compared to the observations.

        Parameters
        ----------
        str_post: jax ndarray
            Array containing a sample from the posterior predictive process.
        obs: jax ndarray
            Array containing the experimental observations over a series of time instances.
        
        Returns
        -------
        nmse: float
            NMSE of the posterior predictive process.
        """ 

        nmse = (100/(obs.var()*len(obs)))*np.sqrt((str_post-obs).T@(str_post-obs))

        return nmse
    
    def exp_util_nmse(self, mcmc, num_samples=2000, set='Test'):

        """
        Returns the NMSE-based expected utility of the decision to employ a particular structural model for the structural deterioration monitoring task, 
        when assessed in terms of the predictive process.

        Parameters
        ----------
        mcmc: Numpyro object
            The inference object that contains the results of the NUTS implementation.
        num_samples: int
            Number of samples from the posterior across different chains to use to calculate the expected utility.
        set: string
            An identifier of the dataset on which to calculate the expected utility. Valid names include 'Train' and 'Test'. Defaults to 'Test'.
        
        Returns
        -------
        exp_util: float
            NMSE-based expected utility.
        """
        
        if (set == 'Train'):
            times = self.tr_times
            obs = self.tr_obs
        else:
            times = self.tst_times
            obs = self.tst_obs

        post = mcmc.get_samples()

        if (self.det_mod_id == 'Logistic'):
            post_dt = self.det_model(times, jnp.expand_dims(post[r'$\alpha$'], -1)[:num_samples], jnp.expand_dims(post[r'$\beta$'], -1)[:num_samples], \
                                     jnp.expand_dims(post[r'$\gamma$'], -1)[:num_samples])
            
        else:
            post_dt = self.det_model(times, jnp.expand_dims(post[r'$\alpha$'], -1)[:num_samples], jnp.expand_dims(post[r'$\beta$'], -1)[:num_samples])

        util = 0

        for i in range(post_dt.shape[0]):
            
            nmse = self.nmse(self.fwd_model.forward(e=self.ym, dt=post_dt[i, :]), obs=obs)
            util += 1 -(1/100)*nmse
        
        exp_util = util/post_dt.shape[0]

        return exp_util

    def logl(self, str_post, sig_post, obs):

        """
        Returns the log-likelihood of the posterior predictive process (strain response), obtained through the posterior thickness loss process and the structural model, 
        compared to the observations and accounting for the prediction error estimate.

        Parameters
        ----------
        str_post: jax ndarray
            Array containing a sample from the posterior predictive process.
        sig_post: jax ndarray
            Array containing a sample from the prediction error standard deviation posterior.
        obs: jax ndarray
            Array containing the experimental observations over a series of time instances.
        
        Returns
        -------
        logl: float
            Log-likelihood of the strain response posterior.
        """ 

        logl = jax.scipy.stats.multivariate_normal.logpdf(x=str_post - obs, mean=jnp.zeros(len(obs)), \
                                                          cov=jnp.identity(len(obs))*sig_post)

        return logl
    
    def exp_util_logl(self, mcmc, num_samples=2000, set='Test'):

        """
        Returns the log-likelihood of the posterior predictive process (strain response), obtained through the posterior thickness loss process and the structural model, 
        compared to the observations and accounting for the prediction error estimate.

        Parameters
        ----------
        mcmc: Numpyro object
            The inference object that contains the results of the NUTS implementation.
        num_samples: int
            Number of samples from the posterior across different chains to use to calculate the expected utility.
        set: string
            An identifier of the dataset on which to calculate the expected utility. Valid names include 'Train' and 'Test'. Defaults to 'Test'.
        
        Returns
        -------
        exp_util: float
            Likelihood-based expected utility of the strain response posterior.
        """

        if (set == 'Train'):
            times = self.tr_times
            obs = self.tr_obs
        else:
            times = self.tst_times
            obs = self.tst_obs

        if (self.mod_id == 'M3'):
            post = self.run_mcmc(oracle=True).get_samples()
        else:    
            post = mcmc.get_samples()

        if (self.det_mod_id == 'Logistic'):
            post_dt = self.det_model(times, jnp.expand_dims(post[r'$\alpha$'], -1)[:num_samples], jnp.expand_dims(post[r'$\beta$'], -1)[:num_samples], \
                                     jnp.expand_dims(post[r'$\gamma$'], -1)[:num_samples])
            
        else:
            post_dt = self.det_model(times, jnp.expand_dims(post[r'$\alpha$'], -1)[:num_samples], jnp.expand_dims(post[r'$\beta$'], -1)[:num_samples])

        post_sig = jnp.expand_dims(post[r'$\sigma$'], -1)[:num_samples]

        logl_map = self.logl(self.fwd_model.forward(e=self.ym, dt=jnp.mean(post_dt, axis=0)), obs.std(), obs)

        util = 0

        for i in range(post_dt.shape[0]):
            
            logl = self.logl(self.fwd_model.forward(e=self.ym, dt=post_dt[i, :]), post_sig[i], obs)
            xval = jnp.abs((logl - logl_map)/jnp.maximum(jnp.abs(logl_map), jnp.abs(logl)))
            util += 1 - xval

        exp_util = (util/post_dt.shape[0]).item()

        return exp_util

    def exp_util_pf(self, mcmc, num_samples=2000, set='Test'):

        """
        Returns the expected utility of the decision to employ a particular structural model for the structural deterioration monitoring task, when assessed in terms of the 
        structural reliability estimate at a specific time instance compared to an oracle model.

        Parameters
        ----------
        mcmc: Numpyro object
            The inference object that contains the results of the NUTS implementation.
        num_samples: int
            Number of samples from the posterior across different chains to use to calculate the expected utility.
        set: string
            An identifier of the dataset on which to calculate the expected utility. Valid names include 'Train' and 'Test'. Defaults to 'Test'.
        
        Returns
        -------
        exp_util: float
            Reliability-based expected utility.
        """
        
        if (set == 'Train'):
            times = self.tr_times
        else:
            times = self.tst_times

        oracle = StructDetMon(model='M3', det_model=self.det_mod_id)
        post_orc = oracle.run_mcmc(oracle=True).get_samples()

        if (self.mod_id == 'M3'):
            post = self.run_mcmc(oracle=True).get_samples()
        else:    
            post = mcmc.get_samples()

        if (self.det_mod_id == 'Logistic'):
            post_dt = self.det_model(times, jnp.expand_dims(post[r'$\alpha$'], -1)[:num_samples], jnp.expand_dims(post[r'$\beta$'], -1)[:num_samples], \
                                     jnp.expand_dims(post[r'$\gamma$'], -1)[:num_samples])
            post_dt = post_dt[:, -1]
            post_dt_orc = self.det_model(times, jnp.expand_dims(post_orc[r'$\alpha$'], -1)[:num_samples], jnp.expand_dims(post_orc[r'$\beta$'], -1)[:num_samples], \
                                     jnp.expand_dims(post_orc[r'$\gamma$'], -1)[:num_samples])
            post_dt_orc = post_dt_orc[:, -1]
            
        else:
            post_dt = self.det_model(times, jnp.expand_dims(post[r'$\alpha$'], -1)[:num_samples], jnp.expand_dims(post[r'$\beta$'], -1)[:num_samples])
            post_dt = post_dt[:, -1]
            post_dt_orc = self.det_model(times, jnp.expand_dims(post_orc[r'$\alpha$'], -1)[:num_samples], jnp.expand_dims(post_orc[r'$\beta$'], -1)[:num_samples])
            post_dt_orc = post_dt_orc[:, -1]

        svm_orc = jnp.zeros(post_dt_orc.shape[0])
        svm_post = jnp.zeros(post_dt.shape[0])

        for i in range(post_dt_orc.shape[0]):

            svm_post = svm_post.at[i].set(self.svm_model.forward(e=self.ym, dt=post_dt[i]))
            svm_orc = svm_orc.at[i].set(oracle.svm_model.forward(e=self.ym, dt=post_dt_orc[i]))

        pf_pred = np.log10(self.pf_samp(svm_post))
        pf_orc = np.log10(self.pf_samp(svm_orc))
        pf_max = np.maximum(np.abs(pf_pred), np.abs(pf_orc))

        exp_util = (1 - (pf_orc - pf_pred)/pf_max).mean()

        return exp_util
    
    def comb_util(self, mcmc, set='Test', w1=0.5, w2=0.5):

        """
        Weighted expected utility of the decision to employ a particular structural model for the structural deterioration monitoring task.

        Parameters
        ----------
        mcmc: Numpyro object
            The inference object that contains the results of the NUTS implementation.
        set: string
            An identifier of the dataset on which to calculate the expected utility. Valid names include 'Train' and 'Test'. Defaults to 'Test'.
        w1: float
            Relative weight of the data-based expected utility.
        w2: float
            Relative weight of the reliability-based expected utility.
        
        Returns
        -------
        comb_util: float
            Combined expected utility.
        """

        return w1*self.exp_util_logl(mcmc, set=set) + w2*self.exp_util_pf(mcmc, set=set)

    def plot_post_det_process(self, mcmc):

        """
        Plotting function that plots the posterior deterioration (thickness loss) process.

        Parameters
        ----------
        mcmc: Numpyro object
            The inference object that contains the results of the NUTS implementation.       
        """         

        post_summary = self.post_summary(mcmc)
        post_dt_mean = post_summary[0]
        post_dt_hpdi = post_summary[1]

        fig, ax = plt.subplots(1, 1, dpi=150)

        ax.plot(self.tr_times, post_dt_mean, color='darkorange', label='Posterior process mean', linestyle='dashed')
        ax.fill_between(self.tr_times, post_dt_hpdi[0], post_dt_hpdi[1], alpha=0.3, facecolor='orange', label='95 \% C.I. (Predicted Process)')

        ax.set_xlabel('$t$ (min)')
        ax.set_ylabel(r'$ \Delta \tau $ [mm]')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.legend(frameon=False, loc='upper left')
        plt.show()
    
    def plot_tr_str_post(self, mcmc):

        """
        Plotting function that plots the predictive process (strain response) over the training set.

        Parameters
        ----------
        mcmc: Numpyro object
            The inference object that contains the results of the NUTS implementation.       
        """         

        post_summary = self.post_summary(mcmc)
        post_str_mean = post_summary[2]
        post_str_hpdi = post_summary[3]
        post_sig_mean = post_summary[4]
        
        fig, ax = plt.subplots(1, 1, dpi=150)
        ax.plot(self.tr_times, post_str_mean, color='darkgreen', label='Predicted Strain (Posterior Mean)', linestyle='dashed')
        ax.fill_between(self.tr_times, post_str_hpdi[0] - 1.96*post_sig_mean, post_str_hpdi[1] + 1.96*post_sig_mean, \
                        alpha=0.3, facecolor='lightgreen', label='95 \% C.I. (Predicted Strain Posterior)')

        ax.scatter(self.tr_times, self.tr_obs, color='darkmagenta', s=2., label=r'Strain Observations', alpha=0.4)
        ax.set_xlabel('$t$ (min)')
        ax.set_ylabel(r'$ \varepsilon_{\mathrm{xx}} \ [\mathrm{\mu\epsilon}]$ ')
        ax.set_xlim(left=(self.tr_times.min()-10), right=(self.tr_times.max()+10))
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.legend(frameon=False)
        plt.show()
    
    def plot_post_pred(self, mcmc, num_samples=1000):

        """
        Plotting function that plots the posterior predictive process (strain response) over the entire timeseries.

        Parameters
        ----------
        mcmc: Numpyro object
            The inference object that contains the results of the NUTS implementation.       
        num_samples: int
            Number of samples to draw from the posterior predictive process.
        """         

        post_preds = self.post_predicitve(mcmc, num_samples=num_samples)
        times = jnp.concatenate([self.tr_times, self.tst_times], axis=0)
        str_obs = jnp.concatenate([self.tr_obs, self.tst_obs], axis=0)

        fig, ax = plt.subplots(1, 1, dpi=150)
        ax.plot(times, jnp.mean(post_preds, axis=0), color='darkgreen', label='Posterior predictive mean', linestyle='dashed')
        ax.fill_between(times, hpdi(post_preds, 0.95)[0], hpdi(post_preds, 0.95)[1], alpha=0.3, facecolor='lightgreen', label='95 \% C.I. (Posterior predictive)')
        ax.scatter(times, str_obs, color='darkmagenta', s=2., label=r'Strain Observations', alpha=0.4)
        ax.set_xlabel('$t$ (min)')
        ax.set_ylabel(r'$ \varepsilon_{\mathrm{xx}} \ [\mathrm{\mu\epsilon}]$ ')
        ax.set_xlim(left=(times.min()-10), right=(times.max()+10))
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.legend(frameon=False)
        plt.show()


def plot_exp_util_nmse(task, det_model='Logistic'):

    """
    Plotting function that plots a bar chart comparing the NMSE-based expected utilities of the different models across different tasks.

    Parameters
    ----------
    task: string
        The task on which we want to compare models. Valid names include 'BMU', 'Diagnosis' and 'Prognosis'.
    det_model: string
        The deterioration model used for the structural deterioration monitoring task. Valid names include 'Logistic' and 'Exponential'.
    """         

    tasks = {'BMU': [BMU('M1'), BMU('M2'), BMU('M3'), BMU('M4'), BMU('M5')], \
             'Diagnosis': [DamageDiag('M1'), DamageDiag('M2'), DamageDiag('M3'), DamageDiag('M4'), DamageDiag('M5')], \
                'Prognosis': [StructDetMon('M1', det_model=det_model), StructDetMon('M2', det_model=det_model), StructDetMon('M3', det_model=det_model), \
                              StructDetMon('M4', det_model=det_model), StructDetMon('M5', det_model=det_model)]}
    utils = []

    for i in range(len(tasks[task])):

        mcmc = tasks[task][i].run_mcmc()
        utils.append(tasks[task][i].exp_util_nmse(mcmc))
        print(utils[i])
    
    util_dict = {'$\mathcal{M}^{(1)}$': utils[0], '$\mathcal{M}^{(2)}$': utils[1], '$\mathcal{M}^{(3)}$': utils[2], \
                    '$\mathcal{M}^{(4)}$': utils[3], '$\mathcal{M}^{(5)}$': utils[4]}
    xvals = list(util_dict.keys())
    yvals = list(util_dict.values())

    plt.figure(dpi=100)
    plt.bar(xvals, yvals, color='tomato', alpha=0.4)
    plt.xlabel('Model compared to oracle $\mathcal{M}^{(3)}$')
    plt.ylabel('Expected utility')
    plt.ylim(bottom=0, top=1.05)
    plt.show()

def plot_exp_util_logl(task, det_model='Logistic'):

    """
    Plotting function that plots a bar chart comparing the likelihood-based expected utilities of the different models across different tasks.

    Parameters
    ----------
    task: string
        The task on which we want to compare models. Valid names include 'BMU', 'Diagnosis' and 'Prognosis'.
    det_model: string
        The deterioration model used for the structural deterioration monitoring task. Valid names include 'Logistic' and 'Exponential'.
    """

    tasks = {'BMU': [BMU('M1'), BMU('M2'), BMU('M3'), BMU('M4'), BMU('M5')], \
             'Diagnosis': [DamageDiag('M1'), DamageDiag('M2'), DamageDiag('M3'), DamageDiag('M4'), DamageDiag('M5')], \
                'Prognosis': [StructDetMon('M1', det_model=det_model), StructDetMon('M2', det_model=det_model), StructDetMon('M3', det_model=det_model), \
                              StructDetMon('M4', det_model=det_model), StructDetMon('M5', det_model=det_model)]}
    utils = []

    for i in range(len(tasks[task])):

        mcmc = tasks[task][i].run_mcmc()
        utils.append(tasks[task][i].exp_util_logl(mcmc))
        print(utils[i])
    
    util_dict = {'$\mathcal{M}^{(1)}$': utils[0], '$\mathcal{M}^{(2)}$': utils[1], '$\mathcal{M}^{(3)}$': utils[2], \
                    '$\mathcal{M}^{(4)}$': utils[3], '$\mathcal{M}^{(5)}$': utils[4]}
    xvals = list(util_dict.keys())
    yvals = list(util_dict.values())

    plt.figure(dpi=100)
    plt.bar(xvals, yvals, color='tomato', alpha=0.4)
    plt.xlabel('Model compared to oracle $\mathcal{M}^{(3)}$')
    plt.ylabel('Expected utility')
    plt.ylim(bottom=0, top=1)
    plt.show()

def plot_comb_util(task, w1=0.5, w2=0.5, det_model='Logistic'):
    
    """
    Plotting function that plots a bar chart comparing the combined (weighted) expected utilities of the different models across different tasks.

    Parameters
    ----------
    task: string
        The task on which we want to compare models. Valid names include 'BMU', 'Diagnosis' and 'Prognosis'.
    w1: float
        Relative weight of the data-based expected utility.
    w2: float
        Relative weight of the reliability-based expected utility.
    det_model: string
        The deterioration model used for the structural deterioration monitoring task. Valid names include 'Logistic' and 'Exponential'.
    """

    tasks = {'Diagnosis': [DamageDiag('M1'), DamageDiag('M2'), DamageDiag('M3'), DamageDiag('M4'),DamageDiag('M5')], \
             'Prognosis': [StructDetMon('M1', det_model=det_model), StructDetMon('M2', det_model=det_model), StructDetMon('M3', det_model=det_model), \
                           StructDetMon('M4', det_model=det_model), StructDetMon('M5', det_model=det_model)]}
    logl_utils = []
    pf_utils = []

    for i in range(len(tasks[task])):

        mcmc = tasks[task][i].run_mcmc()
        logl_utils.append(tasks[task][i].exp_util_logl(mcmc))
        pf_utils.append(tasks[task][i].exp_util_pf(mcmc))
        print(w1*logl_utils[i] + w2*pf_utils[i])
    
    logl_utils_dict = {'$\mathcal{M}^{(1)}$': w1*logl_utils[0], '$\mathcal{M}^{(2)}$': w1*logl_utils[1], '$\mathcal{M}^{(3)}$': w1*logl_utils[2], \
                    '$\mathcal{M}^{(4)}$': w1*logl_utils[3], '$\mathcal{M}^{(5)}$': w1*logl_utils[4]}
    
    pf_utils_dict = {'$\mathcal{M}^{(1)}$': w2*pf_utils[0], '$\mathcal{M}^{(2)}$': w2*pf_utils[1], '$\mathcal{M}^{(3)}$': w2*pf_utils[2], \
                    '$\mathcal{M}^{(4)}$': w2*pf_utils[3], '$\mathcal{M}^{(5)}$': w2*pf_utils[4]}
    
    comb_utils_dict = {'$\mathcal{M}^{(1)}$': w2*pf_utils[0] + w1*logl_utils[0], '$\mathcal{M}^{(2)}$': w2*pf_utils[1] + w1*logl_utils[1], \
                       '$\mathcal{M}^{(3)}$': w2*pf_utils[2] + w1*logl_utils[2], '$\mathcal{M}^{(4)}$': w2*pf_utils[3] + w1*logl_utils[3], \
                        '$\mathcal{M}^{(5)}$': w2*pf_utils[4] + w1*logl_utils[4]}
    
    xvals = jnp.arange(0, 5, 1)
    yvals_logl = list(logl_utils_dict.values())
    yvals_pf = list(pf_utils_dict.values())
    yvals_comb = list(comb_utils_dict.values())

    width = 0.25

    plt.figure(dpi=100)
    plt.bar(xvals, yvals_logl, width=width, color='indigo', alpha=1.0, label=r'$\mathcal{U}_{\mathcal{L}}(\mathcal{M}^{(j)})$')
    plt.bar(xvals + width, yvals_pf, width=width, color='cornflowerblue', alpha=0.4, label=r'$\mathcal{U}_{P_{\mathrm{f}}}(\mathcal{M}^{(j)})$')
    plt.bar(xvals + 2*width, yvals_comb, width=width, color='tomato', alpha=0.8, label=r'$\mathcal{U}(\mathcal{M}^{(j)})$')
    plt.xticks(xvals + width, list(logl_utils_dict.keys()))
    plt.xlabel('Model compared to oracle $\mathcal{M}^{(3)}$')
    plt.ylabel('Expected utility')
    plt.ylim(bottom=0, top=1.1)
    plt.legend(frameon=False, loc='upper left', ncol=3)
    plt.show()

def plot_comp_full(task):

    """
    Plotting function that plots KDE-based densities of the posterior QoIs for the BMU and diagnostic tasks.

    Parameters
    ----------
    task: string
        The task on which we want to compare models. Valid names include 'BMU' and 'Diagnosis'.
    """

    tasks = {'BMU': [BMU('M1'), BMU('M2'), BMU('M3'), BMU('M4'), BMU('M5')], 'Diagnosis': [DamageDiag('M1'), DamageDiag('M2'), DamageDiag('M3'), DamageDiag('M4'), DamageDiag('M5')]}
    labels = {'BMU': [r'$E_{\mathrm{MAP}}$', '\sigma_{\mathrm{MAP}}'], 'Diagnosis': [r'$\Delta \tau_{\mathrm{MAP}}$', '\sigma_{\mathrm{MAP}}']}
    ax_labels = {'BMU': [r'$E \ \mathrm{[GPa]}$', r'$\sigma \ [\mathrm{\mu \epsilon}]$'], 'Diagnosis': [r'$\Delta \tau \ \mathrm{[mm]}$', r'$\sigma \ [\mathrm{\mu \epsilon}]$']}
    units = {'BMU': '$GPa$', 'Diagnosis': 'mm'}
    mcmc = []

    for i in range(len(tasks[task])):

        mcmc.append(tasks[task][i].run_mcmc())

    fig, axes = plt.subplots(1, 2, figsize=(14,6), dpi=100)

    sns.kdeplot(data=tasks[task][0].get_posterior(mcmc[0])[list(tasks[task][0].get_posterior(mcmc[0]).keys())[0]], label='Posterior - $\mathcal{M}^{(1)}$', ax=axes[0], fill=True, color='tomato')
    sns.kdeplot(data=tasks[task][1].get_posterior(mcmc[1])[list(tasks[task][1].get_posterior(mcmc[1]).keys())[0]], label='Posterior - $\mathcal{M}^{(2)}$', ax=axes[0], fill=True, color='orchid')
    sns.kdeplot(data=tasks[task][2].get_posterior(mcmc[2])[list(tasks[task][2].get_posterior(mcmc[2]).keys())[0]], label='Posterior - $\mathcal{M}^{(3)}$', ax=axes[0], fill=True, color='limegreen')
    sns.kdeplot(data=tasks[task][3].get_posterior(mcmc[3])[list(tasks[task][3].get_posterior(mcmc[3]).keys())[0]], label='Posterior - $\mathcal{M}^{(4)}$', ax=axes[0], fill=True, color='darkorange')
    sns.kdeplot(data=tasks[task][4].get_posterior(mcmc[4])[list(tasks[task][4].get_posterior(mcmc[4]).keys())[0]], label='Posterior - $\mathcal{M}^{(5)}$', ax=axes[0], fill=True, color='cornflowerblue')
    axes[0].axvline(tasks[task][0].get_maps(mcmc[0])[list(tasks[task][0].get_posterior(mcmc[0]).keys())[0]], label=labels[task][0] + '$ = %.2f \ - \mathcal{M}^{(1)}$' \
                     % tasks[task][0].get_maps(mcmc[0])[list(tasks[task][0].get_posterior(mcmc[0]).keys())[0]], linestyle='dotted', color='tomato')
    axes[0].axvline(tasks[task][1].get_maps(mcmc[1])[list(tasks[task][1].get_posterior(mcmc[1]).keys())[0]], label=labels[task][0] + '$ = %.2f \ - \mathcal{M}^{(2)}$' \
                     % tasks[task][1].get_maps(mcmc[1])[list(tasks[task][1].get_posterior(mcmc[1]).keys())[0]], linestyle='dotted', color='orchid')
    axes[0].axvline(tasks[task][2].get_maps(mcmc[2])[list(tasks[task][2].get_posterior(mcmc[2]).keys())[0]], label=labels[task][0] + '$ = %.2f \ - \mathcal{M}^{(3)}$' \
                    % tasks[task][2].get_maps(mcmc[2])[list(tasks[task][2].get_posterior(mcmc[2]).keys())[0]], linestyle='dotted', color='limegreen')
    axes[0].axvline(tasks[task][3].get_maps(mcmc[3])[list(tasks[task][3].get_posterior(mcmc[3]).keys())[0]], label=labels[task][0] + '$ = %.2f \ - \mathcal{M}^{(4)}$' \
                    % tasks[task][3].get_maps(mcmc[3])[list(tasks[task][3].get_posterior(mcmc[3]).keys())[0]], linestyle='dotted', color='darkorange')
    axes[0].axvline(tasks[task][4].get_maps(mcmc[4])[list(tasks[task][4].get_posterior(mcmc[4]).keys())[0]], label=labels[task][0] + '$ = %.2f \ - \mathcal{M}^{(5)}$' \
                    % tasks[task][4].get_maps(mcmc[4])[list(tasks[task][4].get_posterior(mcmc[4]).keys())[0]], linestyle='dotted', color='cornflowerblue')
    axes[0].set_xlabel(ax_labels[task][0])
    axes[0].set_ylabel(r'Density')
    axes[0].legend(frameon=False)

    sns.kdeplot(data=tasks[task][0].get_posterior(mcmc[0])[list(tasks[task][0].get_posterior(mcmc[0]).keys())[1]], label='Posterior - $\mathcal{M}^{(1)}$', ax=axes[1], fill=True, color='tomato')
    sns.kdeplot(data=tasks[task][1].get_posterior(mcmc[1])[list(tasks[task][1].get_posterior(mcmc[1]).keys())[1]], label='Posterior - $\mathcal{M}^{(2)}$', ax=axes[1], fill=True, color='orchid')
    sns.kdeplot(data=tasks[task][2].get_posterior(mcmc[2])[list(tasks[task][2].get_posterior(mcmc[2]).keys())[1]], label='Posterior - $\mathcal{M}^{(3)}$', ax=axes[1], fill=True, color='limegreen')
    sns.kdeplot(data=tasks[task][3].get_posterior(mcmc[3])[list(tasks[task][3].get_posterior(mcmc[3]).keys())[1]], label='Posterior - $\mathcal{M}^{(4)}$', ax=axes[1], fill=True, color='darkorange')
    sns.kdeplot(data=tasks[task][4].get_posterior(mcmc[4])[list(tasks[task][4].get_posterior(mcmc[4]).keys())[1]], label='Posterior - $\mathcal{M}^{(5)}$', ax=axes[1], fill=True, color='cornflowerblue')
    axes[1].axvline(tasks[task][0].get_maps(mcmc[0])[list(tasks[task][0].get_posterior(mcmc[0]).keys())[1]], label='$' + labels[task][1] + '= %.2f \ - \mathcal{M}^{(1)}$' \
                    % tasks[task][0].get_maps(mcmc[0])[list(tasks[task][0].get_posterior(mcmc[0]).keys())[1]], linestyle='dotted', color='orchid')
    axes[1].axvline(tasks[task][1].get_maps(mcmc[1])[list(tasks[task][1].get_posterior(mcmc[1]).keys())[1]], label='$' + labels[task][1] + '= %.2f \ - \mathcal{M}^{(2)}$' \
                    % tasks[task][1].get_maps(mcmc[1])[list(tasks[task][1].get_posterior(mcmc[1]).keys())[1]], linestyle='dotted', color='orchid')
    axes[1].axvline(tasks[task][2].get_maps(mcmc[2])[list(tasks[task][2].get_posterior(mcmc[2]).keys())[1]], label='$' + labels[task][1] + '= %.2f \ - \mathcal{M}^{(3)}$' \
                     % tasks[task][2].get_maps(mcmc[2])[list(tasks[task][2].get_posterior(mcmc[2]).keys())[1]], linestyle='dotted', color='limegreen')
    axes[1].axvline(tasks[task][3].get_maps(mcmc[3])[list(tasks[task][3].get_posterior(mcmc[3]).keys())[1]], label='$' + labels[task][1] + '= %.2f \  - \mathcal{M}^{(4)}$' \
                    % tasks[task][3].get_maps(mcmc[3])[list(tasks[task][3].get_posterior(mcmc[3]).keys())[1]], linestyle='dotted', color='darkorange')
    axes[1].axvline(tasks[task][4].get_maps(mcmc[4])[list(tasks[task][4].get_posterior(mcmc[4]).keys())[1]], label='$' + labels[task][1] + '= %.2f \ - \mathcal{M}^{(5)}$' \
                    % tasks[task][4].get_maps(mcmc[4])[list(tasks[task][4].get_posterior(mcmc[4]).keys())[1]], linestyle='dotted', color='cornflowerblue')
    axes[1].legend(frameon=False)
    axes[1].set_xlabel(ax_labels[task][1])
    axes[1].set_ylabel(r'Density')

    plt.tight_layout(pad=2)
    plt.show()

def plot3dscan():

    my_mesh = mesh.Mesh.from_file(r'D:\Ph.D. Thesis\Research Work\Main Project\3PB - Corrosion Experiment\3D Scanning\NTUA-2023-11-15-plakaki HR.stl')
    xs = my_mesh.vectors[:, :, 0].flatten()
    ys = my_mesh.vectors[:, :, 1].flatten()
    zs = my_mesh.vectors[:, :, 2].flatten()

    fig, ax = plt.subplots()
    cm = plt.cm.rainbow
    im = ax.scatter(xs, ys, c=zs.max()-zs, cmap=cm, s=2.)
    cbar = plt.colorbar(im)
    # cbar.ax.tick_params(labelsize=20)
    cbar.set_label(label=r'$\Delta \tau$ (mm)', size=12)
    ax.set_xlabel('$x$ (mm)', fontname='Times New Roman')
    ax.set_ylabel('$y$ (mm)', fontname='Times New Roman')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.savefig(r'D:\Ph.D. Thesis\Research Work\Main Project\Journal Papers\Mech. Syst. Signal Process. 02-24\Paper Manuscript\3d_Scan_Surf.png', format='png', bbox_inches='tight', dpi=600)
    # plt.show()

# This is an example main method. Modify it to produce any results you want to see by instantiating any of the classes and calling any of their methods!
# At its current iteration it either plots the parameter posteriors and trace plots for the structural deterioration monitoring task when using the intermediate fidelity solid model (M2) and an exponential detarioration model,
# or the bar chart of weighted expected utilities for different models for the same task and deterioration model. Comment the corresponding line out to get the desired result! =

# Defining the main
def main():

    plot_exp_util_logl('BMU')

# # Running the main
if __name__ == "__main__":

    main()  
