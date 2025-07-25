import numpy as np
import math
import lmfit as lm
from matplotlib import pyplot as plt
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
import other
import copy
import emcee
import corner
from multiprocessing import Pool
import os
import pickle
from rebound_oc_delay import simulate_oc_delay
import matplotlib.animation as animation

d2sec = 24*60*60
_LN_2PI = np.log(2.0 * np.pi)

class OC_model:
    """
    A class to manage observational data, including epoch analysis, model component management, 
    and data fitting.

    Attributes:
        epochs (np.ndarray): An array of epochs.
        model_components (list): A list of model component objects.
        Ref_mintime (float): The reference minimum time used for calculations.
        Ref_period (float): The reference period used for calculations.
        name (str): Name identifier for the OC_model.
        nan_policy (str): Policy for handling NaN values ('raise', etc.).
    
    Methods:
        add_model_component(model_component): 
            Adds a new model component to the model_components list, ensuring name uniqueness.
        remove_model_component(by_object=None, by_index=None, by_name=None): 
            Removes a model component based on specified criteria.
        calculate_oc(epochs=None) -> np.ndarray: 
            Computes the observational (O-C) data based on the current model components.
        plot(show=True): 
            Plots the calculated O-C data against epochs.
        summary(return_type="dict"): 
            Returns a summary of the model components and model name.
        save_model(filename): 
            Saves the model object to a file.
        load_model(filename): 
            Loads a model object from a file.
    """

    def __init__(self, epochs=np.array([]), mintimes=([]), mintypes=([]), 
                 name="OC_Model", model_components: list = None, 
                 Ref_mintime=None, Ref_period=None, nan_policy='raise',
                 m1=None, m2=None, inc=None):
        """
        Initializes a new instance of OC_model.

        Parameters:
            epochs (np.ndarray): Optional. An array of epochs (default is an empty array).
            name (str): Optional. Name identifier for the model (default "OC_Model").
            model_components (list): Optional. A list of model component objects. If None, defaults to an empty list.
            Ref_mintime (float): Optional. The reference minimum time used in the model.
            Ref_period (float): Optional. The reference period used in the model.
            nan_policy (str): Optional. Policy for handling NaN values (default is 'raise').
        """
        self.name = name
        # If no model components are provided, initialize with an empty list.
        self.model_components = model_components if model_components is not None else [] 
        # Ensure epochs is stored as a NumPy array.
        self.epochs = np.array(epochs) if epochs is not None else np.array([])
        self.Ref_mintime = Ref_mintime
        self.Ref_period = Ref_period
        self.nan_policy = nan_policy
        self.m1 = m1
        self.m2 = m2
        self.inc = inc

    def add_model_component(self, model_component):
        """
        Adds a new model component to the list of model_components, ensuring that its name is unique.

        Parameters:
            model_component (object): The model component object to be added.

        Raises:
            ValueError: If a model component with the same name already exists.
        """
        # Warn if a similar type of model component already exists (except for specific allowed types)
        if any(isinstance(existing_model_component, type(model_component)) 
               for existing_model_component in self.model_components) \
           and not isinstance(model_component, LiTE) \
           and not isinstance(model_component, LiTE_abspar):
            print(f"Warning: model_component of type {model_component.__class__.__name__} already exists but new one added anyways")

        # Check for unique name among the existing model components.
        existing_names = [existing_model_component.name for existing_model_component in self.model_components]
        if model_component.name in existing_names:
            raise ValueError("model_component with same name already exists in OC_Model. "
                             "Use 'object.name = desired name' to change the component name.")

        # Append the new model component.
        self.model_components.append(model_component)
        # Set the reference parameters in the new component.
        model_component.Ref_period = self.Ref_period
        model_component.Ref_mintime = self.Ref_mintime

    def remove_model_component(self, by_object=None, by_index=None, by_name=None):
        """
        Removes a model component from the list based on the specified criterion.
        Only one parameter should be provided.

        Parameters:
            by_object (object): The model component object to remove.
            by_index (int): The index of the model component in the list to remove.
            by_name (str): The name of the model component to remove.

        Raises:
            ValueError: If more than one parameter is provided or if no parameter is specified.
            TypeError: If the type of by_index or by_name is incorrect.
        """
        # Create a list of parameters provided.
        parameters = [by_object, by_index, by_name]
        # Count how many of the parameters are not None.
        not_none_params = sum(p is not None for p in parameters)

        # Ensure only one removal criterion is provided.
        if not_none_params > 1:
            raise ValueError("Only one of by_object, by_index, or by_name should be provided.")
        
        if by_object is not None:
            # Remove by matching the component's name.
            for mc in self.model_components:
                if mc.name == by_object.name:
                    self.model_components.remove(mc)
                    break
        elif by_index is not None:
            if not isinstance(by_index, int):
                raise TypeError("Index must be an integer.")
            self.model_components.pop(by_index)
        elif by_name is not None:
            if not isinstance(by_name, str):
                raise TypeError("Name must be a string.")
            if by_name not in [model_component.name for model_component in self.model_components]:
                raise ValueError("Model component with specified name not found in the model_components list.")
            # Filter out the component with the matching name.
            self.model_components = [model_component for model_component in self.model_components if model_component.name != by_name]
        else:
            raise ValueError("One of by_object, by_index, or by_name must be specified.")

    def calculate_oc(self, epochs=None) -> np.ndarray:
        """
        Calculates the observational O-C (Observed minus Calculated) values by summing the contributions
        from each model component over the provided epochs.

        Parameters:
            epochs (np.ndarray, optional): An array of epochs to use for the calculation. 
                If None, the object's stored epochs are used.

        Returns:
            np.ndarray: The calculated O-C values as an array.
        """
        equation = 0  # Initialize the cumulative equation.
        # Loop through each model component.
        for model_component in self.model_components:
            # Ensure that the reference parameters are set for each component.
            if model_component.Ref_period is None:
                model_component.Ref_period = self.Ref_period
            if model_component.Ref_mintime is None:
                model_component.Ref_mintime = self.Ref_mintime

            # Use provided epochs or fall back to object's epochs.
            ep = self.epochs if epochs is None else epochs
            # Sum the individual model's contribution.
            equation += model_component.individual_model(ep)

        return equation

    def plot(self, show=True):
        """
        Plots the calculated O-C data against the epochs.

        Parameters:
            show (bool): If True, displays the plot immediately (default is True).
        """
        plt.plot(self.epochs, self.calculate_oc(), label=self.name)
        plt.xlabel("Epoch")
        plt.ylabel("OC (d)")
        plt.legend()
        if show:
            plt.show()

    def summary(self, print=False, return_type="dict"):
        """
        Returns a summary of the OC_model, including its name and the names of its model components.

        Parameters:
            return_type (str): Type of summary to return. Options are "dict" or "dataframe" (default: "dict").

        Returns:
            dict or pandas.DataFrame: A summary of the model components.
        
        Raises:
            ValueError: If return_type is not 'dict' or 'dataframe'.
        """
        sum = {
            "Model Name": self.name,
            "Model Components": [model_component.name for model_component in self.model_components]
        }
        if return_type == "dict":
            return sum
        elif return_type == "dataframe":
            return pd.DataFrame(sum, index=[0])
        else:
            raise ValueError("return_type must be 'dict' or 'dataframe'")

    def save_model(self, filename):
        """
        Saves the current OC_model instance to a file using pickle.

        Parameters:
            filename (str): The path to the file where the model should be saved.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filename):
        """
        Loads an OC_model instance from a pickle file.

        Parameters:
            filename (str): The path to the file from which to load the model.

        Returns:
            OC_model: The loaded OC_model instance.
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)
        
    def fix_units(self):
        """
        Fixes the units of the model components and parameters.

        This method iterates over the model components and parameters, converting them to the
        """
        model_unit_fixed = copy.deepcopy(self)
        for component in model_unit_fixed.model_components:
            for param_name, param_obj in component.params.items():
                if param_name != "T_LiTE":
                    param_obj.value = _unit_conv(
                        param_obj.value, param_obj.unit, component._main_units[param_name],
                        ref_period=self.Ref_period, ref_mintime=self.Ref_mintime,
                        parameter_name=param_name, JD_convertable=param_obj.JD_convertable
                    )
                    if param_obj.std is not None:
                        param_obj.std = _unit_conv(
                            param_obj.std, param_obj.unit, component._main_units[param_name],
                            ref_period=self.Ref_period, ref_mintime=self.Ref_mintime,
                            parameter_name=param_name, JD_convertable=param_obj.JD_convertable
                        )
                    if param_obj.min is not None:
                        param_obj.min = _unit_conv(
                            param_obj.min, param_obj.unit, component._main_units[param_name],
                            ref_period=self.Ref_period, ref_mintime=self.Ref_mintime,
                            parameter_name=param_name, JD_convertable=param_obj.JD_convertable
                        )
                    if param_obj.max is not None:
                        param_obj.max = _unit_conv(
                            param_obj.max, param_obj.unit, component._main_units[param_name],
                            ref_period=self.Ref_period, ref_mintime=self.Ref_mintime,
                            parameter_name=param_name, JD_convertable=param_obj.JD_convertable
                        )
                else:
                    param_obj.value = _unit_conv(
                        param_obj.value, param_obj.unit, component._main_units[param_name],
                        ref_period=self.Ref_period, ref_mintime=self.Ref_mintime,
                        parameter_name=param_name, JD_convertable=param_obj.JD_convertable
                        )
                    
                    if param_obj.std is not None:
                        if param_obj.unit == "BJD" and isinstance(component, LiTE):
                            param_obj.std = _unit_conv(
                                param_obj.std, "day", component._main_units[param_name],
                                ref_period=self.Ref_period, ref_mintime=self.Ref_mintime,
                                parameter_name=param_name, JD_convertable=param_obj.JD_convertable
                            )
                        else:
                            param_obj.std = _unit_conv(
                                param_obj.std, param_obj.unit, component._main_units[param_name],
                                ref_period=self.Ref_period, ref_mintime=self.Ref_mintime,
                                parameter_name=param_name, JD_convertable=param_obj.JD_convertable
                            )
                    if param_obj.min is not None:
                        param_obj.min = _unit_conv(
                            param_obj.min, param_obj.unit, component._main_units[param_name],
                            ref_period=self.Ref_period, ref_mintime=self.Ref_mintime,
                            parameter_name=param_name, JD_convertable=param_obj.JD_convertable
                        )
                    if param_obj.max is not None:
                        param_obj.max = _unit_conv(
                            param_obj.max, param_obj.unit, component._main_units[param_name],
                            ref_period=self.Ref_period, ref_mintime=self.Ref_mintime,
                            parameter_name=param_name, JD_convertable=param_obj.JD_convertable
                        )
        return model_unit_fixed
    
    def fix_units_reversed(self):
        """
        Converts parameters from main units to the specified unit.
        """
        model_unit_reversed = copy.deepcopy(self)
        for component in model_unit_reversed.model_components:
            for param_name, param_obj in component.params.items():
                param_obj.value = _unit_conv(
                    param_obj.value, component._main_units[param_name], param_obj.unit,
                    ref_period=self.Ref_period, ref_mintime=self.Ref_mintime,
                    parameter_name=param_name, JD_convertable=param_obj.JD_convertable
                )
                if param_obj.std is not None:
                    param_obj.std = _unit_conv(
                        param_obj.std, component._main_units[param_name], param_obj.unit,
                        ref_period=self.Ref_period, ref_mintime=self.Ref_mintime,
                        parameter_name=param_name, JD_convertable=param_obj.JD_convertable
                    )
                if param_obj.min is not None:
                    param_obj.min = _unit_conv(
                        param_obj.min, component._main_units[param_name], param_obj.unit,
                        ref_period=self.Ref_period, ref_mintime=self.Ref_mintime,
                        parameter_name=param_name, JD_convertable=param_obj.JD_convertable
                    )
                if param_obj.max is not None:
                    param_obj.max = _unit_conv(
                        param_obj.max, component._main_units[param_name], param_obj.unit,
                        ref_period=self.Ref_period, ref_mintime=self.Ref_mintime,
                        parameter_name=param_name, JD_convertable=param_obj.JD_convertable
                    )
        return model_unit_reversed
    
    def to_oc_data(self, Ref_mintime, Ref_period, error_sigma=.2, object_name=None):
        """
        Generate an OC_data instance containing observation times and O–C values
        predicted by this model, adding Gaussian noise with a given sigma.

        Parameters
        ----------
        Ref_mintime : float
            Reference minimum time (T₀) to build observation times.
        Ref_period : float
            Reference period (P₀) per epoch.
        error_sigma : float, optional
            Standard deviation of Gaussian noise to add to the simulated times.
            If zero (default), no noise is added.
        object_name : str, optional
            Name to assign to the created OC_data object.
            If None, falls back to this OC_model’s own name attribute.

        Returns
        -------
        OC_data
            An OC_data object populated with synthetic O–C data based on this model,
            with Gaussian errors of standard deviation `error_sigma`.
        """
        # 1) Round epochs to nearest integer
        epochs = np.round(np.atleast_1d(self.epochs).flatten()).astype(int)

        has_abspar = False
        for i in self.model_components:
            if isinstance(i, LiTE_abspar):
                has_abspar = True
                break
        if not has_abspar:
            oc_values = self.calculate_oc(epochs)
            true_times = Ref_mintime + epochs * Ref_period + oc_values
            mintype_flags = np.zeros(true_times.shape, dtype=int)
            errors = np.full(true_times.shape, error_sigma)

        else:
            temp_times = Ref_mintime + epochs * Ref_period
            mintype_flags = np.zeros(temp_times.shape, dtype=int)
            errors = np.full(temp_times.shape, error_sigma)
            temp_data = OC_data(
                        object_name = object_name or self.name,
                        Mintimes    = temp_times,
                        Mintypes    = mintype_flags,
                        Errors      = errors,
                        Ref_mintime = Ref_mintime,
                        Ref_period  = Ref_period)
            temp_model = copy.deepcopy(self)
            for mc in temp_model.model_components:
                for p, v in mc.params.items():
                    v.vary=False
            temp_fit = fit(data=temp_data, model=temp_model)
            if self.m1 is None or self.m2 is None or self.inc is None:
                raise ValueError("m1, m2 and inc needed to create oc_data")
            temp_oc = temp_fit.total_oc_delay([], self.m1, self.m2, self.inc, Ecorr=None)
            true_times = temp_times + temp_oc


        noise = np.random.normal(loc=0.0, scale=error_sigma, size=true_times.shape)
        observation_times = true_times + noise
        
        dg = ['created_data'] * len(observation_times)
        

        # 7) Assemble and return the OC_data
        new_data = OC_data(
            object_name = object_name or self.name,
            Mintimes    = observation_times,
            Mintypes    = mintype_flags,
            Errors      = errors,
            Ref_mintime = Ref_mintime,
            Ref_period  = Ref_period,
            Data_group  = dg
        )
        new_data.m1 = self.m1
        new_data.m2 = self.m2
        new_data.inc = self.inc

        return new_data
    
class OC_data:
    """
    A class for calculating time for corrected orbital phase.

    Attributes:
        Mintimes (numpy.ndarray): Array of minimum times.
        Mintypes (numpy.ndarray): Array of minimum types '0' for primary '1' for secondary.
        Errors (numpy.ndarray): Array of errors.
        Units (numpy.ndarray): Array of units.
        Ref_mintime (float): Reference minimum time.
        Ref_period (float): Reference period.
        data_file (str): Path to data file.

    Methods:
        generate_oc: Generates corrected orbital phase.
        remove_outliers: Removes outliers from the dataset.
        sigma_outliers: Detects outliers using sigma-clipping method.
        zscore_outliers: Detects outliers using z-score method.
        box_outliers: Detects outliers using boxplot method.
        chauvenet_outliers: Detects outliers using Chauvenet's criterion method.
        manual_elimination: Allows manual elimination of data points.
        generate_oc: Generates the O-C (Observed minus Calculated) data.
        plot_OC: Plots the orbital phase correction (OC).
    """
    def __init__(
        self,
        object_name="undefined",
        Mintimes=None,
        Mintypes=None,
        Errors=None,
        Units=None,
        Ref_mintime=None,
        Ref_period=None,
        Data_group=None,
        Weights=None,
        data_file=None
    ):
        import numpy as np
        import pandas as pd

        self.object_name = object_name
        self._calculated = False
        self.binned = False
        self.m1 = None
        self.m2 = None
        self.inc = None

        # 1) If a data_file is provided, read everything from it and skip parameter overrides
        if data_file is not None:
            self._read_data_file(data_file)

        else:
            # 2) Otherwise, use the passed-in arrays (or defaults)
            self.Mintimes   = np.atleast_1d(Mintimes) if Mintimes   is not None else np.array([])
            self.Mintypes   = np.atleast_1d(Mintypes).astype(int) if Mintypes is not None else np.array([], dtype=int)
            self.Errors     = np.atleast_1d(Errors)   if Errors     is not None else np.full(self.Mintimes.shape, 0.0001)
            self.Units      = np.atleast_1d(Units)    if Units      is not None else np.full(self.Mintimes.shape, "Default")
            self.Data_group = np.atleast_1d(Data_group) if Data_group is not None else np.full(self.Mintimes.shape, "None")
            self.Ref_mintime = Ref_mintime
            self.Ref_period   = Ref_period
            self.Weights      = np.atleast_1d(Weights) if Weights is not None else None

        # 3) If weights weren’t provided or loaded as empty, fill defaults
        if not hasattr(self, 'Weights') or self.Weights is None or self.Weights.size == 0:
            self.fill_weights()

        # 4) Generate Ecorr and OC only if Mintimes/Mintypes etc. are set and valid
        if (
            hasattr(self, 'Mintimes')
            and self.Mintimes.size > 0
            and hasattr(self, 'Mintypes')
            and self.Mintypes.size == self.Mintimes.size
            and hasattr(self, 'Ref_mintime')
            and self.Ref_mintime is not None
            and hasattr(self, 'Ref_period')
            and self.Ref_period is not None
        ):
            self.Ecorr, self.OC = self.generate_oc()
        else:
            self.Ecorr = np.array([])
            self.OC    = np.array([])

        # 5) Build the DataFrame for easy access
        self.df = pd.DataFrame({
            'Mintimes':   self.Mintimes,
            'Mintypes':   self.Mintypes,
            'Errors':     self.Errors,
            'Units':      self.Units,
            'Data_group': self.Data_group,
            'Ecorr':      self.Ecorr,
            'OC':         self.OC,
            'Weights':    self.Weights
        })
        
    def __setattr__(self, name, value) -> None:
        if name == "df":
            for col in value.columns:
                super().__setattr__(col, value[col].values)
                super().__setattr__("df", value)
        elif hasattr(self, "df"):
            if name in self.df.columns:
                self.df[name] = value
        super().__setattr__(name, value)
        
        if name in ["Mintimes", "Mintypes", "Ref_mintime", "Ref_period"] and \
        hasattr(self, 'Mintimes') and hasattr(self, 'Mintypes') and \
        hasattr(self, 'Ref_mintime') and hasattr(self, 'Ref_period'):
            if len(self.Mintimes) > 0 and len(self.Mintypes) > 0 and \
            self.Ref_mintime is not None and self.Ref_period is not None:
                if not self._calculated:
                    self.Ecorr, self.OC = self.generate_oc()
                    
    def remove_data(self,
                    values_x_min=None,
                    values_x_max=None,
                    values_y_min=None,
                    values_y_max=None,
                    data_groups=None):
        """
        Return a copy of the data with measurements inside the specified rectangle
        or matching groups removed.

        Parameters:
            values_x_min (float): Lower Ecorr bound of region to delete.
            values_x_max (float): Upper Ecorr bound of region to delete.
            values_y_min (float): Lower O–C bound of region to delete.
            values_y_max (float): Upper O–C bound of region to delete.
            data_groups (str or list): Group label(s) to remove entirely.
        """
        # deep‐copy so we don’t mutate the original
        new_data = copy.deepcopy(self)
        df = new_data.df

        # remove points inside the x‐range
        if values_x_min is not None and values_x_max is not None:
            # drop rows where Ecorr is between x_min and x_max (inclusive)
            df = df[~((df["Ecorr"] >= values_x_min) & (df["Ecorr"] <= values_x_max))]
        elif values_x_min is not None:
            # drop rows where Ecorr is >= x_min
            df = df[~(df["Ecorr"] >= values_x_min)]
        elif values_x_max is not None:
            # drop rows where Ecorr is <= x_max
            df = df[~(df["Ecorr"] <= values_x_max)]

        # remove points inside the y‐range
        if values_y_min is not None and values_y_max is not None:
            # drop rows where OC is between y_min and y_max (inclusive)
            df = df[~((df["OC"] >= values_y_min) & (df["OC"] <= values_y_max))]
        elif values_y_min is not None:
            # drop rows where OC is >= y_min
            df = df[~(df["OC"] >= values_y_min)]
        elif values_y_max is not None:
            # drop rows where OC is <= y_max
            df = df[~(df["OC"] <= values_y_max)]

        # remove entire data groups if requested
        if data_groups is not None:
            if isinstance(data_groups, str):
                df = df[df["Data_group"] != data_groups]
            elif isinstance(data_groups, list):
                df = df[~df["Data_group"].isin(data_groups)]

        new_data.df = df
        return new_data
            
    def fill_weights(self, value=None):
        """
        Populate or reset the Weights array for fitting based on Errors or constant value.

        Parameters:
            value (float, optional): If provided, all weights set to this constant.
        """
        if value == None:
            self.Weights = 1/self.Errors**2
        else:
            weights = value

    def _read_data_file(self, data_file):
        """
        Reads data from a CSV or Excel file and assigns values to class attributes.

        Args:
            data_file (str): Path to the data file.

        Raises:
            ValueError: If the file type is unsupported.
        """
        if data_file.endswith((".csv", ".xls", ".xlsx")):
            data = pd.read_csv(data_file, skiprows=1) if data_file.endswith(".csv") else pd.read_excel(data_file, skiprows=1)
        else:
            raise ValueError("Unsupported file type. Use 'csv' or 'xlsx' instead")
        if "Mintimes" in data.columns and (~data["Mintimes"].isna().all()):
            self.Mintimes = data["Mintimes"].values
            setattr(self, "Mintypes", data["Mintypes"].values if data["Mintypes"].notna().any() else np.zeros_like(self.Mintimes) if data["Mintimes"].notna().any() else None)
            setattr(self, "Errors", data["Errors"].values if data["Errors"].notna().any() else np.zeros_like(self.Mintimes)+1 if data["Mintimes"].notna().any() else None)
            setattr(self, "Data_group", data["Data Group"].values if data["Data Group"].notna().any() else np.zeros_like(self.Mintimes)+1 if data["Mintimes"].notna().any() else None)
            setattr(self, "Units", data["Units"].values if data["Units"].notna().any() else np.full(self.Mintimes.shape, "Default") if data["Mintimes"].notna().any() else None)
        elif "Mintimes" in data.columns and "O-C" in data.columns and data["Mintimes"].isna().all() and not data["O-C"].isna().all():
            self.Ecorr = data["Ecorr"].values
            self.OC = data["O-C"].values

        first_cell = pd.read_excel(data_file, usecols='A', nrows=1, header=None).iloc[0, 0]
        key_value_pairs = first_cell.split(',')
        
        for pair in key_value_pairs:
            key, value = pair.split('=')
            setattr(self, key, float(value))
            
    def remove_outliers(self, outliers):
        """
        Removes outliers from the dataset.

        Args:
            outliers (numpy.ndarray): Boolean array indicating outliers.

        Returns:
            None
        """
        if outliers is None:
            return
        
        new_data = copy.deepcopy(self)
        new_data.df = self.df[~outliers]
        return new_data
    
    def filter_data_range(self, dtype=None, min_value=-np.inf, max_value=np.inf, eq=None, noteq=None, plot=False):
        """
        Filter data by column name and range, or equality conditions.

        Parameters:
            dtype (str): Column in df to filter.
            min_value (float): Lower bound.
            max_value (float): Upper bound.
            eq (any): Exact match filter.
            noteq (any): Exclusion filter.
            plot (bool): Show outlier plot if True.
        Returns:
            OC_data: Filtered instance.
        """
        if dtype is None:
            raise ValueError("dtype must be provided")
        if min_value == -np.inf and max_value == np.inf:
            if eq is None and noteq is None:
                raise ValueError("Either min_value or max_value or eq or noteq must be provided")
            elif eq is not None:
                if isinstance(eq, str):
                    new_df = self.df[self.df[dtype] == eq]
                    eliminated = self.df[self.df[dtype] != eq]
                elif isinstance(eq, int):
                    new_df = self.df[self.df[dtype] == eq]
                    eliminated = self.df[self.df[dtype] != eq]
                elif isinstance(eq, list):
                    new_df = self.df[self.df[dtype].isin(eq)]
                    eliminated = self.df[~self.df[dtype].isin(eq)]
            elif noteq is not None:
                if isinstance(noteq, str):
                    new_df = self.df[self.df[dtype] != noteq]
                    eliminated = self.df[self.df[dtype] == noteq]
                elif isinstance(noteq, int):
                    new_df = self.df[self.df[dtype] != noteq]
                    eliminated = self.df[self.df[dtype] == noteq]
                elif isinstance(noteq, list):
                    new_df = self.df[~self.df[dtype].isin(noteq)]
                    eliminated = self.df[self.df[dtype].isin(noteq)]
        else:
            new_df = self.df[(self.df[dtype] >= min_value) & (self.df[dtype] <= max_value)]
            eliminated = self.df[(self.df[dtype] < min_value) | (self.df[dtype] > max_value)]
        new_data = copy.deepcopy(self)
        new_data.df = new_df
        return new_data
    
    def filter_data_df(self, mask):
        """
        Filters the DataFrame using a custom boolean mask.

        Args:
            mask (pandas.Series or numpy.ndarray): Boolean mask for filtering.

        Returns:
            A new instance of the class with filtered data.
        """
        new_data = copy.deepcopy(self)
        new_data.df = self.df[mask]
        return new_data
        
    def _plot_outliers(self, outliers, v_lines=[], h_lines=[]):
        """
        Plots outliers on a bigger figure.

        Parameters
        ----------
        outliers : boolean array
            Mask of which points are outliers.
        v_lines : list of floats, optional
            x-positions at which to draw vertical limit lines.
        h_lines : list of floats, optional
            y-positions at which to draw horizontal limit lines.
        figsize : tuple, optional
            Figure size in inches, defaults to (12, 6).
        """
        import matplotlib.pyplot as plt

        # Prepare data
        Ecorr_outlier = self.Ecorr[outliers]
        OC_outlier    = self.OC[outliers]
        Ecorr_res     = self.Ecorr[~outliers]
        OC_res        = self.OC[~outliers]

        # Create figure & axes with custom size
        fig, ax = plt.subplots(figsize=(14,8))

        # Draw limit lines (if any)
        v_lines = v_lines or []
        for v in v_lines:
            ax.axvline(v, color="r", linestyle="--", label="Vertical Limits")
        h_lines = h_lines or []
        for h in h_lines:
            ax.axhline(h, color="r", linestyle="--", label="Horizontal Limits")

        # Plot outliers vs. the rest
        if len(Ecorr_outlier) > 0:
            ax.plot(Ecorr_outlier, OC_outlier, "rx", label="Outliers")
        ax.plot(Ecorr_res, OC_res, "b.", label="Inliers")

        # Labels & legend
        ax.set_xlabel("Cycles")
        ax.set_ylabel("O–C")
        ax.legend()

        plt.show()
        
    def sigma_outliers(self, threshold=3, plot=False, OC=None, additional_method = None, additional_params = None):
        """
        Identifies outliers using sigma-clipping method.

        Args:
            threshold (float): Number of standard deviations to consider as threshold.
            plot (bool): Whether to plot outliers.
            OC (numpy.ndarray): Array of observed minus calculated (O-C) values.
            additional_method (str): Additional method for outlier detection. Can be 'moving_window', 'fixed_window' or 'spline'.
            additional_params (dict): Additional parameters for additional_method.

        Returns:
            numpy.ndarray: Boolean array indicating outliers.
        """
        if additional_method is not None:
            if additional_method not in ['moving_window', 'fixed_window', 'spline']:
                raise ValueError("Invalid additional method. Choose 'moving_window', 'fixed_window' or 'spline'.")
            outliers = self._addm(additional_method, additional_params, main="sigma_outliers", main_params=[threshold], plot=plot)
            return outliers
        OC = self.OC if OC is None else OC
        mean_oc = np.mean(OC)
        std_oc = np.std(OC)
        condition = (OC < mean_oc + threshold*std_oc) & (OC > mean_oc - threshold*std_oc)
        if plot:
            self._plot_outliers(~condition, h_lines=[mean_oc + threshold*std_oc, mean_oc - threshold*std_oc])
        return ~condition
    
    def zscore_outliers(self, threshold=3, plot=False, OC=None, additional_method = None, additional_params = None):
        """
        Identifies outliers using z-score method.

        Args:
            threshold (float): Number of standard deviations to consider as threshold.
            plot (bool): Whether to plot outliers.
            OC (numpy.ndarray): Array of observed minus calculated (O-C) values.
            additional_method (str): Additional method for outlier detection. Can be 'moving_window, 'fixed_window' or 'spline'.
            additional_params (dict): Additional parameters for additional_method.

        Returns:
            numpy.ndarray: Boolean array indicating outliers.
        """
        from scipy.stats import zscore
        if additional_method is not None:
            outliers = self._addm(additional_method, additional_params, main="zscore_outliers", main_params=[threshold], plot=plot)
            return outliers
        OC = self.OC if OC is None else OC
        z_scores = zscore(OC)
        condition = np.abs(z_scores) < threshold
        if plot:
            self._plot_outliers(~condition)
        return ~condition
    
    def box_outliers(self, plot=False, threshold=1.5, OC=None, additional_method = None, additional_params = None):
        """
        Identifies outliers using boxplot method.

        Args:
            plot (bool): Whether to plot outliers.
            threshold (float): Multiplier for the interquartile range to consider as threshold.
            OC (numpy.ndarray): Array of observed minus calculated (O-C) values.
            additional_method (str): Additional method for outlier detection. Can be 'moving_window, 'fixed_window' or 'spline'.
            additional_params (dict): Additional parameters for additional_method.

        Returns:
            numpy.ndarray: Boolean array indicating outliers.
        """
        if additional_method is not None:
            outliers = self._addm(additional_method, additional_params, main="zscore_outliers", main_params=[threshold], plot=plot)
            return outliers
        OC = self.OC if OC is None else OC
        q1 = np.percentile(OC, 25)
        q3 = np.percentile(OC, 75)
        iqr = q3 - q1
        lower_threshold = q1 - threshold * iqr
        upper_threshold = q3 + threshold * iqr
        outliers = np.logical_or(OC < lower_threshold, OC > upper_threshold)
        if plot:
            self._plot_outliers(outliers, h_lines=[lower_threshold, upper_threshold])
        return outliers
    
    def chauvenet_outliers(self, plot=False, threshold=0.5, OC=None, additional_method = None, additional_params = None):
        """
        Identifies outliers using Chauvenet's criterion method.

        Args:
            plot (bool): Whether to plot outliers.
            threshold (float): Threshold probability for Chauvenet's criterion.
            OC (numpy.ndarray): Array of observed minus calculated (O-C) values.
            additional_method (str): Additional method for outlier detection.
            additional_params (dict): Additional parameters for additional_method.

        Returns:
            numpy.ndarray: Boolean array indicating outliers.
        """
        if additional_method is not None:
            outliers = self._addm(additional_method, additional_params, main="zscore_outliers", main_params=[threshold], plot=plot)
            return outliers
        from scipy import stats as st
        OC = self.OC if OC is None else OC
        mean = OC.mean()
        std = OC.std()
        d = ((OC - mean) / std)
        possibility = st.norm.cdf(d)
        criteria = (1 - possibility) * len(OC)
        outliers = criteria < threshold
        if plot:
            self._plot_outliers(outliers)
        return outliers

    def _addm(self, additional_method, additional_params, main, main_params, plot=False):
        """
        Handles additional parameter situation for outlier finding methods
        """
        main = getattr(self, main)
        ad_p = ["window_rate", "window_step_rate", "window_size", "window_count", "smoothing", "degree", "window_start"]
        for i in ad_p:
            if i not in additional_params:
                additional_params[i] = None
        if additional_method is not None:
            if additional_method == "moving_window":
                outliers = self._moving_window(window_rate=additional_params["window_rate"], window_step_rate=additional_params["window_step_rate"], window_size=additional_params["window_size"], method=main, window_start=additional_params["window_start"], threshold=main_params[0], plot=plot)
            elif additional_method == "fixed_window":
                outliers = self._fixed_window(window_rate=additional_params["window_rate"], window_size=additional_params["window_size"], window_count=additional_params["window_count"], window_start=additional_params["window_start"], method=main, threshold=main_params[0], plot=plot)
            elif additional_method == "spline":
                outliers = self._spline(smoothing=additional_params["smoothing"], degree=additional_params["degree"], method=main, threshold=main_params[0], plot=plot)
        return outliers

    def _moving_window(self, window_rate=.1, window_step_rate=.01, window_size=None, method="sigma_outliers", window_start=None, threshold=3, plot=False):
        """
        Handles moving_window parameter situation for outlier finding methods
        Creates a window and slide it along the graph. Finds outliers for per slide.
        Args:
            window_rate(float): window's rate to whole graph.
            window_step_rate(float): move rate of window to whole graph.
            window_size(float): window size as.
            window_start(float): window starting point.
        """
        if window_size is not None:
            window_rate = window_size / (max(self.Ecorr) - min(self.Ecorr))
        outliers = np.array([])
        dif = max(self.Ecorr) - min(self.Ecorr)
        window_start = min(self.Ecorr) if window_start is None else window_start
        window_end = dif * window_rate
        while window_start < max(self.Ecorr):
            OC_window = self.OC[(self.Ecorr >= window_start) & (self.Ecorr <= window_end)]
            outliers = np.concatenate((outliers, OC_window[method(threshold=threshold, OC=OC_window)]))
            window_start += dif * window_step_rate
            window_end += dif * window_step_rate
        outliers = np.unique(outliers)
        outliers = np.isin(np.arange(len(self.OC)), np.where(np.isin(self.OC, outliers)))
        if plot:
            self._plot_outliers(outliers)
        return outliers
    
    def _fixed_window(self, window_rate=.1, window_size=None, window_count=None, window_start=None,  method="sigma_outliers", threshold=3, plot=False):
        """
        Handles fixed_window parameter situation for outlier finding methods
        creates fixed windows along the graph. Finds outliers for per window.
        Args:
            window_rate(float): window's rate to whole graph.
            window_size(float): window size as.
            window_start(float): window starting point.
        """
        if window_size is not None:
            window_rate = window_size / (max(self.Ecorr) - min(self.Ecorr))
        elif window_count is not None:
            window_rate = 1 / (window_count -1)
        outliers = np.array([])
        spacing = (max(self.Ecorr) - min(self.Ecorr)) * window_rate
        window_start = min(self.Ecorr)-spacing/2 if window_start is None else window_start
        split_points = np.arange(window_start, self.Ecorr.max()+spacing, spacing)
        OC_list = []
        for i in range(len(split_points) - 1):
            start = split_points[i]
            end = split_points[i + 1]
            OC_list.append(self.OC[(self.Ecorr >= start) & (self.Ecorr < end)])
        for array in OC_list:
            outliers = np.concatenate((outliers, array[method(threshold=threshold, OC=array)]))
        outliers = np.isin(np.arange(len(self.OC)), np.where(np.isin(self.OC, outliers)))
        if plot:
            for i in split_points:
                plt.axvline(i, alpha=.3, linestyle="--")
            self._plot_outliers(outliers)
        return outliers

    def _spline(self, smoothing=1, degree=3, method="sigma_outliers", threshold=3, plot=False):
        """
        Handles spline parameter situation for outlier finding methods
        Creates spline. Get residual from spline and uses given method to find outliers.
        Args:
            smoothing(float): smoothing parameter for spline.
            degree(int): degree of spline.
        """
        from scipy import interpolate
        Ecorr_unique = self.Ecorr
        OC_unique = self.OC
        for i in range(len(self.Ecorr)):
            Ecorr_unique[i] += i

        tck = interpolate.splrep(Ecorr_unique, OC_unique, s=smoothing, k=degree)
        Ecorr_interpolated = Ecorr_unique
        oc_interpolated = interpolate.splev(Ecorr_interpolated, tck, der=0)

        OC_dif = OC_unique - oc_interpolated

        outliers = method(threshold=threshold, OC=OC_dif)

        if plot:
            self._plot_outliers(outliers)
            plt.subplot(2, 1, 1)
            plt.plot(Ecorr_interpolated, oc_interpolated, "r-")
            plt.plot(Ecorr_unique, OC_unique, "b.")
            plt.xlabel('Ecorr')
            plt.ylabel('OC')
            plt.subplot(2, 1, 2)
            plt.plot(Ecorr_interpolated, OC_dif, "y.")
            plt.plot(Ecorr_interpolated[outliers], OC_dif[outliers], "rx")
            plt.show()
        return outliers
    
    def binning(self, start_x=None, end_x=None, method="average", plot=False, group=None, bin_count=1, smart_bin=False, smart_bin_period=10, binned_group_name=None):
        """
        Returns weighted mean of data within the given range, divided into specified number of bins.

        Args:
            start_x (float): Minimum x value.
            end_x (float): Maximum x value.
            method (str): Method to use for binning ('average' or 'median').
            plot (bool): Whether to plot the result.
            group (str or list): Data group to filter on.
            bin_count (int): Number of bins to divide the data into. TODO add methods to automatic calculate bin count
            smart_bin (bool): It does seperate data with more gap than smart_bin_period. If there are more or equal gaps than bin_count it can create bins than bin_count.
            smart_bin_period (float): If smart_bin is True, it will seperate data with more gap than this value.

        Returns:
            DataProcessor: New DataProcessor instance with the binned data.
        """
        def weighted_mean(df, weights):
            mask = ~np.isnan(weights) & (weights != None)
            oc = df["OC"][mask]
            weights = weights[mask]
            ecorr = df["Ecorr"][mask].values
            weighted_oc = np.average(oc, weights=weights)
            weighted_Ecorr = np.average(ecorr)
            sum_weights = np.sum(weights)
            weighted_error = 1 / np.sqrt(sum_weights)
            return weighted_Ecorr, weighted_oc, weighted_error

        def median(df, weights):
            median_oc = np.median(df["OC"])
            median_Ecorr = np.average(df["Ecorr"])
            sum_weights = np.sum(weights)
            weighted_error = 1 / np.sqrt(sum_weights)
            return median_Ecorr, median_oc, weighted_error

        if group is not None and (start_x is None or end_x is None):
            df = self.df[self.df["Data_group"]==group]
            start_x = min(df["Ecorr"])
            end_x = max(df["Ecorr"])


        self.df = self.df.sort_values(by="Ecorr")
        if isinstance(bin_count, str) and bin_count.lower() == "freedman":
            rn = max(self.df["OC"]) - min(self.df["OC"])
            q1 = np.percentile(self.df["OC"], 25)
            q3 = np.percentile(self.df["OC"], 75)
            length = len(self.df["OC"].values)
            IQR = q3 - q1
            bin_width = 2 * IQR / length**(1/3)
            bin_count = int(rn / bin_width)
            
        weights = 1 / self.df["Errors"] ** 2
        mask = (self.df["Ecorr"] >= start_x) & (self.df["Ecorr"] <= end_x)

        if group is not None:
            if isinstance(group, str):
                mask &= (self.df["Data_group"] == group)
            elif isinstance(group, list):
                mask &= self.df["Data_group"].isin(group)

        df = self.df[mask]
        weights = weights[mask]

        if df.empty:
            print("No data found in the specified range and group to bin.")
            return self
        
        gap_starts = []
        gap_ends = []
        gap_starts.append(df["Ecorr"].min())
        if smart_bin:
            gaps = np.diff(df["Ecorr"].values)
            gap_count = np.sum(gaps > smart_bin_period)
            for i, gap in enumerate(gaps):
                if gap > smart_bin_period:
                    gap_ends.append(df["Ecorr"].values[i])
                    gap_starts.append(df["Ecorr"].values[i + 1])
            if gap_count >= bin_count:
                bin_count = gap_count + 1
        gap_ends.append(df["Ecorr"].max())
        
        bin_lengths = np.array(gap_ends) - np.array(gap_starts)
        bin_distribution = bin_lengths / np.sum(bin_lengths) * bin_count
        bin_dist_int = bin_distribution.astype(int)
        zero_count = np.sum(bin_dist_int == 0)

        bin_dist_int[bin_dist_int == 0] = 1
        
        remaining_bins = bin_count - sum(bin_dist_int)
        if remaining_bins < 0:
            for i in range(-remaining_bins):
                leftovers = bin_distribution%1
                leftovers_more_1 = leftovers[bin_dist_int > 1]
                left_ind = np.argwhere(leftovers == min(leftovers_more_1))
                bin_dist_int[left_ind] -= 1
                bin_distribution[left_ind] =.999999999
                remaining_bins += 1
        
        for i in range(int(remaining_bins)):
            leftovers = bin_distribution%1
            left_ind = np.argwhere(leftovers == max(leftovers))
            bin_dist_int[left_ind] += 1
            bin_distribution[left_ind] = 0
            remaining_bins -= 1
        
        
        binned_data = []
        for i in range(len(gap_starts)):
            first_element = gap_starts[i]
            last_element = gap_ends[i]
            bins_for_gap = np.linspace(first_element, last_element, bin_dist_int[i] + 1)
        
            for j in range(len(bins_for_gap)-1):
                bin_mask = (df["Ecorr"] >= bins_for_gap[j]) & (df["Ecorr"] < bins_for_gap[j+1])
                if j == bin_count - 1:  # Include the last element in the last bin
                    bin_mask = (df["Ecorr"] >= bins_for_gap[j]) & (df["Ecorr"] <= bins_for_gap[j+1])
                bin_df = df[bin_mask]
                bin_weights = weights[bin_mask]

                if bin_df.empty:
                    continue

                if method == "average":
                    last_Ecorr, Last_oc, last_error = weighted_mean(bin_df, bin_weights)
                elif method == "median":
                    last_Ecorr, Last_oc, last_error = median(bin_df, bin_weights)

                if binned_group_name is not None:
                    new_dg_name = binned_group_name
                else:
                    new_dg_name = f"{group}_binned" if group is not None else "binned"
                last_mintime = last_Ecorr*self.Ref_period + self.Ref_mintime 
                self.binned = True
                new_row = pd.DataFrame({
                    "Mintimes": [last_mintime],
                    "OC": [Last_oc],
                    "Ecorr": [last_Ecorr],
                    "Errors": [last_error],
                    "Data_group": [new_dg_name],
                    "Units": [None]
                })
                binned_data.append(new_row)

        if not binned_data:
            print("No data was binned.")
            return self

        binned_df = pd.concat(binned_data, ignore_index=True)
        df2 = self.df[~mask]
        df2 = pd.concat([df2, binned_df], ignore_index=True)
        df2.sort_values(by="Ecorr", inplace=True)

        if plot:
            self.kpr = False

            def on_key_press(event):
                if event.key == "enter":
                    self.weighted_mean_save(df2)
                    plt.close()
                    self.kpr = True
                elif event.key == " ":
                    self.weighted_mean_save(df2)
                    plt.close()
                    self.kpr = False
                elif event.key == "escape":
                    plt.close()
                    self.kpr = False

            fig, ax = plt.subplots()
            ax.errorbar(df["Ecorr"], df["OC"], fmt='.', color='b')
            ax.text(0.05, 0.95, 'Press "enter" to save and continue, press "space" to save and quit\n'
                                'Press "esc" to cancel and quit',
                    transform=fig.transFigure, va='top')
            fig.canvas.mpl_connect('key_press_event', on_key_press)
            plt.show()
            return self.kpr
        else:
            new_data = copy.deepcopy(self)
            new_data.weighted_mean_save(df2)
            new_data.fill_weights()
            return new_data
        
    def weighted_mean_save(self, df2):
        self.df = df2
                
    def fill_errors(self, method="average", group=None, fill_nan_groups=True, value=1):
        def fill_nan(row):
            if pd.isna(row['Errors']):
                group_name = row['Data_group']
                if group_name in group_errors:
                    return group_errors[group_name]
                else:
                    return np.nan  # Or any other suitable value to handle missing group average??????
            else:
                return row['Errors']
        df = pd.DataFrame({"Errors": self.Errors, "Data_group": self.Data_group})
        # grouping
        if isinstance(group, str) or isinstance(group, float):
            df2 = df[df['Data_group'] == group]
        elif isinstance(group, list):
            df2 = df[df['Data_group'].isin(group)]
        elif group is None:
            df2 = df   
        df2['Errors'] = pd.to_numeric(df['Errors'], errors='coerce')    
        # methods
        if method == "average":
            group_errors = df2.groupby('Data_group')['Errors'].mean()
            if fill_nan_groups:
                overall_error = df['Errors'].mean()
        if method == "median":
            group_errors = df2.groupby('Data_group')['Errors'].median()
            if fill_nan_groups:
                overall_error = df['Errors'].median()
        if method == "max":
            group_errors = df2.groupby('Data_group')['Errors'].max()
            if fill_nan_groups:
                overall_error = df['Errors'].max()
        if method == "with_value":
            group_errors = df2.groupby('Data_group')['Errors'].max()
            overall_error = df['Errors'].mean()
            if group is None:
                df2["Errors"] = value
            elif isinstance(group, str) or isinstance(group, float):
                df2 = df[df['Data_group'] == group]
                df2["Errors"] = value
            elif isinstance(group, list):
                df2 = df[df['Data_group'].isin(group)]
                df2["Errors"] = value
                    
        pd.options.mode.copy_on_write = True
        group_errors = group_errors.apply(lambda x: overall_error if pd.isna(x) else x)
        df2['Errors'] = df2.apply(fill_nan, axis=1)
        df.update(df2)
        new_data = copy.deepcopy(self)
        new_data.Errors = df['Errors'].values
        new_data.df['Errors'] = df['Errors'].values
        return new_data

    def get_weighted_mean_manual(self):
        """
        Launch an interactive selector to manually choose weighting limits.

        This will open a plot window. Follow on-screen prompts to accept or cancel.
        """
        cont = True
        while cont:
            fig, ax = plt.subplots()
            selector = other.Vertical_selector(ax, fig, self.Ecorr, self.OC)
            selector.scatter = ax.plot(self.Ecorr, self.OC, "b.")[0]
            plt.show()

            if selector.save:
                cont = self.simplize(selector.vlines_x[0], selector.vlines_x[1], plot=True)
            else:
                cont = False

    def manual_outliers(self):
        """
        Allows user to manually eliminate outliers.

        Returns:
            numpy.ndarray: Boolean array indicating outliers.
        """
        fig, ax = plt.subplots()
        selector = other.ClickAndDragSelector(ax, fig, self.Ecorr, self.OC)
        selector.scatter = ax.plot(self.Ecorr, self.OC, "b.")[0]

        plt.show()

        if selector.selected is None:
            print("No data removed, If you selected data press 'enter' after selection")
        else:
            return selector.selected
        
    def convert_units(self, desired_unit):
        """
        Convert all Mintimes between HJD and BJD based on RA/DEC.

        Parameters:
            desired_unit (str): 'HJD' or 'BJD'.
        Raises:
            ValueError: If missing coordinate data or unknown units.
        """
        if self.RA is None or self.DEC is None:
            raise ValueError("RA and DEC values needed for unit conversion")
        for unit in self.Units:
            if unit != "HJD" and unit != "BJD":
                raise ValueError("All units should be HJD or BJD")
        if desired_unit == "HJD":
            hjd_list = []
            for i, bjd in enumerate(self.Mintimes):
                if self.Units[i] == "BJD":
                    hjd_list.append(self.bary_to_helio(self.RA, self.DEC, bjd))
                elif self.Units[i] == "HJD":
                    hjd_list.append(bjd)
            self.Mintimes = np.array(hjd_list)
            self.Units = np.full_like(self.Units, "HJD")
        elif desired_unit == "BJD":
            bjd_list = []
            for i, hjd in enumerate(self.Mintimes):
                if self.Units[i] == "HJD":
                    bjd_list.append(self.helio_to_bary(self.RA, self.DEC, hjd))
                elif self.Units[i] == "BJD":
                    bjd_list.append(hjd)
            self.Mintimes = np.array(bjd_list)
            self.Units = np.full_like(self.Units, "BJD")

    @staticmethod
    def helio_to_bary(ra_deg, dec_deg, hjd):
        """
        Convert Heliocentric JD to Barycentric JD for given sky coordinates.
        """
        fix = False
        if hjd < 2400000:
            hjd += 2400000
            fix = True
        helio = Time(hjd, scale='utc', format='jd')
        obs = EarthLocation.of_site('Keck Observatory')
        star = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame='icrs')
        ltt_helio = helio.light_travel_time(star, 'heliocentric', location=obs)
        guess = helio - ltt_helio
        delta = (guess + guess.light_travel_time(star, 'heliocentric', location=obs)).jd - helio.jd
        guess -= delta * u.d
        ltt_bary = guess.light_travel_time(star, 'barycentric', location=obs)
        bjd = (guess.tdb + ltt_bary).jd
        if fix:
            bjd -= 2400000
        return bjd

    @staticmethod
    def bary_to_helio(ra_deg, dec_deg, bjd):
        """
        Convert Barycentric JD to Heliocentric JD for given sky coordinates.
        """
        fix = False
        if bjd < 2400000:
            bjd += 2400000
            fix = True
        bary = Time(bjd, scale='tdb', format='jd')
        obs = EarthLocation.of_site('Keck Observatory')
        star = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame='icrs')
        ltt_bary = bary.light_travel_time(star, 'barycentric', location=obs)
        guess = bary - ltt_bary
        delta = (guess + guess.light_travel_time(star, 'barycentric', location=obs)).jd - bary.jd
        guess -= delta * u.d
        ltt_helio = guess.light_travel_time(star, 'heliocentric', location=obs)
        hjd = (guess.utc + ltt_helio).jd
        if fix:
            hjd -= 2400000
        return hjd
    
    def lomb_scargle(self, min_freq, max_freq, num_frequencies, plot=False):
        """
        Compute Lomb-Scargle periodogram of O-C and return the dominant period.

        Parameters:
            min_freq (float): Minimum frequency to scan.
            max_freq (float): Maximum frequency to scan.
            num_frequencies (int): Number of frequency steps.
            plot (bool): Display the periodogram if True.

        Returns:
            frequencies (ndarray): Array of scanned frequencies.
            power (ndarray): Normalized Lomb–Scargle power.
            peak_freq (float): Frequency with maximum power.
            peak_period (float): Dominant period = 1/peak_freq.
        """
        import numpy as np
        from scipy.signal import lombscargle
        import matplotlib.pyplot as plt

        # 1) build frequency grid and convert to angular freqs
        frequencies = np.linspace(min_freq, max_freq, num_frequencies)
        angular_frequencies = 2 * np.pi * frequencies  

        # 2) compute raw periodogram on (t,y) = (self.Ecorr, self.OC)
        pgram = lombscargle(self.Ecorr, self.OC, angular_frequencies)

        # 3) normalize by N/2 to get comparable power
        power = pgram / (len(self.Ecorr) / 2)

        # 4) find peak frequency and compute its period
        idx_peak   = np.argmax(power)        # index of maximum power
        peak_freq  = frequencies[idx_peak]   # most significant frequency
        peak_period = 1.0 / peak_freq        # corresponding period

        # 5) optional: plot and mark the peak
        if plot:
            plt.plot(frequencies, power, label='Lomb–Scargle')
            plt.axvline(peak_freq, color='r', linestyle='--',
                        label=f'Peak @ {peak_freq:.2e} (1/peak={peak_period:.0f})')
            plt.xlabel('Frequency')
            plt.ylabel('Normalized Power')
            plt.legend()
            plt.show()

        return frequencies, power, peak_freq, peak_period


    # Creating O-C with data
    def generate_oc(self, inplace=False) -> tuple[np.ndarray, np.ndarray]:
        """ 
        Generates the O-C (Observed minus Calculated) data based on provided or existing time measurements, minima types, and orbital parameters.

        Parameters:
        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the corrected epochs (Ecorr) and the O-C values.
        """
        if self.binned:
            raise ValueError("Data is binned, Generated object will not be given true values. If you know what are you doing use 'object.binned = False'")
        
        E = (self.Mintimes - self.Ref_mintime) / self.Ref_period
        Ecorr = np.zeros_like(E)
        round_indices = self.Mintypes == 0
        positive_mask = E >= 0
        Ecorr[round_indices] = np.round([val for val in E[round_indices]])
        Ecorr[positive_mask & ~round_indices] = np.trunc([val for val in E[positive_mask & ~round_indices]]) + 0.5
        Ecorr[~positive_mask & ~round_indices] = np.trunc([val for val in E[~positive_mask & ~round_indices]]) - 0.5
        
        C = self.Ref_mintime + Ecorr * self.Ref_period
        OC = self.Mintimes - C
        self._calculated = True
        if inplace:
            self.Ecorr = Ecorr
            self.OC = OC
        return Ecorr, OC
    
    def plot_OC(self, vertical_lines=[], horizontal_lines=[], show=True, with_errors=True, legend=True):
        """
        Plots the orbital phase correction (OC).

        This function plots the orbital phase correction (OC) against the orbital phase,
        with different colors for each data group.

        Args:
            vertical_lines (list): List of x-values for vertical lines.
            horizontal_lines (list): List of y-values for horizontal lines.
            show (bool): Whether to show the plot.
            with_errors (bool): Whether to include error bars.
            legend (bool): Whether to include a legend.

        Returns:
            None
        """

        # Define a color map
        colors = plt.get_cmap('tab10', len(self.df["Data_group"].unique()))

        # Create a color dictionary for each unique data group
        color_dict = {name: colors(i) for i, name in enumerate(self.df["Data_group"].unique())}

        # Set a larger figure size
        plt.figure(figsize=(14, 8))

        # Plot each group with a different color
        grouped = self.df.groupby("Data_group")
        for name, group in grouped:
            plt.plot(group["Ecorr"], group["OC"], ".", label=name, color=color_dict[name])
            if with_errors:
                # Ensure Errors column has no NaN values before plotting error bars
                valid_errors = group["Errors"].notna()
                plt.errorbar(
                    group["Ecorr"][valid_errors], 
                    group["OC"][valid_errors], 
                    yerr=group["Errors"][valid_errors], 
                    fmt=".", 
                    color=color_dict[name]
                )

        # Add vertical lines if any
        for x in vertical_lines:
            plt.axvline(x=x, color='grey', linestyle='--')

        # Add horizontal lines if any
        for y in horizontal_lines:
            plt.axhline(y=y, color='grey', linestyle='--')
            
        plt.xlabel("Cycle")
        plt.ylabel("O-C")
        
        if legend:
            plt.legend(title="Data Group")

        if show:
            plt.show()
            
    def save_data(self, filename):
        """
        Saves the data to a file using pickle.

        Args:
            filename (str): The name of the file to save the data to.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
          
    @staticmethod  
    def load_data(filename):
        """
        Loads data from a file using pickle.

        Args:
            filename (str): The name of the file to load the data from.
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)

class fit:
    """
    A class to perform model fitting on observational data using various methods,
    including least-squares minimization and Bayesian MCMC sampling.

    Attributes:
        model: The model instance containing model components and reference parameters.
        data: The observational data object (e.g., an OC_data instance).
        samples: Array of samples from the fitting process (if available).
        outfile_tag: Suffix to append to output filenames.
        fitted_model: The updated model instance with fitted parameter values.
    """

    def __init__(self, model, data, integrator="IAS15"):
        """
        Initializes the fit instance.

        Parameters:
            model: The model instance (with a list of model_components).
            data: The observational data object used for fitting.
        """
        self.model = model
        self.data = data
        self.samples = None  
        self.outfile_tag = None
        self.integrator = integrator

    def model_params(self, model=None):
        """
        Return a pandas DataFrame of model parameters, always including uncertainties.

        Parameters
        ----------
        model : OC_model, optional
            The model whose parameters to tabulate (default: self.fitted_model if available, else self.model).

        Returns
        -------
        pandas.DataFrame
            Table with columns ['Component', 'Parameter', 'Value', 'Std', 'Unit'].
        """
        import rebound
        import numpy as np
        import pandas as pd

        # If no model was passed in, use fitted_model if it exists, otherwise self.model
        model = model or getattr(self, "fitted_model", self.model)

        rows = []
        for comp in model.model_components:
            # 1) Add each parameter (name, value, std, unit) as-is
            for pname, pobj in comp.params.items():
                rows.append({
                    "Component": comp.name,
                    "Parameter": pname,
                    "Value": pobj.value,
                    "Std": pobj.std,
                    "Unit": pobj.unit
                })

            # 2) If this component is of type LiTE_abspar, compute the "a" parameter and its uncertainty with Rebound
            if isinstance(comp, LiTE_abspar):
                # --- 2.1) Extract the values of LiTE_abspar parameters ---
                m3_val    = comp.mass.value      # Mass of the third body (Msun)
                sigma_m3  = comp.mass.std        # Uncertainty of m3 (Msun)

                P_val     = comp.P_LiTE.value    # LiTE period (in days)
                sigma_P   = comp.P_LiTE.std      # Uncertainty of P (days)

                e_val     = comp.ecc.value       # Eccentricity
                inc_deg   = comp.inc.value       # Inclination (in degrees)
                omega_deg = comp.omega.value     # Argument of periastron (in degrees)
                T_val     = comp.T_LiTE.value    # Time of periastron (Julian Day)

                # --- 2.2) Convert angles from degrees to radians ---
                inc_rad   = np.radians(inc_deg)
                omega_rad = np.radians(omega_deg)

                # --- 2.3) Set up the Rebound simulation ---
                sim = rebound.Simulation()
                sim.units = ('day', 'AU', 'Msun')   # Units: time=day, distance=AU, mass=Msun
                sim.integrator = "IAS15"
                sim.dt = 0.5

                # 2.3.a) Add the binary as a single central mass (m1 + m2)
                #     We assume self.data.m1, self.data.m2, self.data.m1_std, self.data.m2_std exist
                m1_val   = self.data.m1
                m2_val   = self.data.m2
                sigma_m1 = getattr(self.data, "m1_std", 0.0)
                sigma_m2 = getattr(self.data, "m2_std", 0.0)

                sim.add(m = m1_val + m2_val)

                # 2.3.b) Add the third body; Rebound computes a3 automatically from P=P_val
                sim.add(
                    m     = m3_val,
                    P     = P_val,
                    e     = e_val,
                    inc   = inc_rad,
                    T     = T_val,
                    omega = omega_rad
                )

                # 2.3.c) Shift to center-of-mass reference
                sim.move_to_com()

                # 2.4) The semimajor axis of the third body: .particles[1].a (in AU)
                a3_val = sim.particles[1].a

                # --- 2.5) Propagate uncertainties to get Std(a3) ---
                # Total mass M_tot = m1 + m2 + m3
                M_tot     = m1_val + m2_val + m3_val
                sigma_M   = np.sqrt(sigma_m1**2 + sigma_m2**2 + sigma_m3**2)

                # Fractional uncertainties
                frac_M = sigma_M / M_tot if M_tot != 0 else 0.0
                frac_P = sigma_P / P_val if P_val != 0 else 0.0

                # For Kepler's law a ∝ (M_tot * P^2)^{1/3}, fractional error:
                # Δa / a = (1/3) * sqrt( (ΔM / M_tot)^2 + (2 * ΔP / P)^2 )
                sigma_a3 = a3_val * np.sqrt((frac_M / 3)**2 + ((2 * frac_P) / 3)**2)

                # 2.6) Append the computed a3 value and its uncertainty to the table
                rows.append({
                    "Component": comp.name,
                    "Parameter": "a",
                    "Value": a3_val,
                    "Std": sigma_a3,
                    "Unit": "AU"
                })

        df = pd.DataFrame(rows, columns=["Component", "Parameter", "Value", "Std", "Unit"])
        return df

        

    def save_fit(self, filename):
        """
        Saves the current fit instance to a file using pickle.

        Parameters:
            filename (str): The file path where the fit instance will be saved.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load_fit(filename):
        """
        Loads a fit instance from a file using pickle.

        Parameters:
            filename (str): The file path from which the fit instance will be loaded.

        Returns:
            fit: The loaded fit instance.
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)
        
    def fit_model(self, method='leastsq', iter_cb=None, scale_covar=True, verbose=False, 
                fit_kws=None, nan_policy=None, calc_covar=True, max_nfev=None, 
                coerce_farray=True, integrator="IAS15", ias15_accuracy=1e-8, **kwargs):
        """
        Perform least squares fit (use leastsq as method)
        or nelder-mead fit (use nelder as method)
        on O-C data using lmfit.

        Parameters:
            method (str): Minimization algorithm name. (nelder or leastsq)
            iter_cb (callable): Callback function per iteration.
            scale_covar (bool): Scale the covariance matrix.
            verbose (bool): Print detailed output if True.
            fit_kws (dict): Additional kwargs for lmfit.
            nan_policy (str): NaN handling policy.
            calc_covar (bool): Calculate covariance if True.
            max_nfev (int): Max function evaluations.
            coerce_farray (bool): Force array conversion.
        Returns:
            lmfit.MinimizerResult: Fit result object.
        """
        fit2 = copy.deepcopy(self)
        fit2.model.Ref_period = fit2.data.Ref_period
        fit2.model.Ref_mintime = fit2.data.Ref_mintime
        fit2.integrator=integrator
        fit2.ias15_accuracy = ias15_accuracy
        # Prepare variable parameter names in order.
        variable_param_names = []
        for component in fit2.model.model_components:
            for param_name, param_obj in component.params.items():
                if param_obj.vary:
                    variable_param_names.append((component, param_name))

        # Define the lmfit model function using total_oc_delay.
        def total_oc_delay_lmfit(x, **params):
            # Extract variable parameters from lmfit params (raw, unconverted).
            variable_params = []
            for component, param_name in variable_param_names:
                full_name = f"{param_name}_{component.name}"
                variable_params.append(params[full_name])
            # Call total_oc_delay with the extracted raw parameters.
            return fit2.total_oc_delay(
                variable_params, m1=fit2.data.m1, m2=fit2.data.m2, inc=fit2.data.inc, Ecorr=x, fix_units_first=True, mintimes_in_data=True
            )

        # Assemble lmfit Parameters with **raw values**, no conversion.
        params = lm.Parameters()
        for component in fit2.model.model_components:
            identifier = component.name
            component.Ref_mintime = fit2.model.Ref_mintime
            component.Ref_period = fit2.model.Ref_period
            for attr, value in component.params.items():
                param_name = f'{attr}_{identifier}'
                param_value = getattr(component, attr).value
                param_min = getattr(component, attr).min
                param_max = getattr(component, attr).max
                params.add(param_name, value=param_value, vary=value.vary, 
                        min=param_min, max=param_max, expr=value.expr, 
                        brute_step=value.brute_step)

        # Create lmfit Model using the total_oc_delay wrapper.
        model = lm.Model(total_oc_delay_lmfit, independent_vars=['x'], nan_policy=fit2.model.nan_policy)

        weights = fit2.data.Weights
        if np.any(np.isnan(weights)):
            raise ValueError("Weights cannot contain NaN values. Fix nan values of your data weights.")

        # Perform the fit.
        result = model.fit(
            fit2.data.OC, params, x=fit2.data.Ecorr, weights=weights, 
            method=method, iter_cb=iter_cb, scale_covar=scale_covar, 
            fit_kws=fit_kws, nan_policy=nan_policy, calc_covar=calc_covar, 
            max_nfev=max_nfev, coerce_farray=coerce_farray, **kwargs
        )

        # Update the fitted model.
        self.fitted_model = fit2.create_fit_model(result)
        return result

    def fit_lin(self, inplace=True, plot=False):
        """
        Fit and remove linear trend from O-C residuals.

        Parameters:
            inplace (bool): Update data references if True.
            plot (bool): Plot trend and data if True.
        Returns:
            lmfit.MinimizerResult: Result of linear fit.
        """
        # If plotting, store the original O-C data before any modifications.
        if plot:
            original_OC = self.data.OC.copy()

        # Define a linear function.
        def lin(x, a, b):
            return a * x + b

        # Initialize lmfit parameters with starting values of 1.
        param = lm.Parameters()
        param.add('a', value=1)
        param.add('b', value=1)

        # Build the lmfit Model for the linear function.
        model = lm.Model(lin, independent_vars=['x'])
        result = model.fit(self.data.OC, x=self.data.Ecorr, weights=self.data.Weights, params=param)

        if inplace:
            # Remove the linear trend from the data.
            trend = result.params['a'].value * self.data.Ecorr + result.params['b'].value
            self.data.OC = self.data.OC - trend
            self.data.Ref_mintime = self.data.Ref_mintime + result.params['b'].value
            self.data.Ref_period = self.data.Ref_period + result.params['a'].value

        if plot:
            import matplotlib.pyplot as plt
            x = self.data.Ecorr
            # Compute the fitted line.
            fitted_line = result.params['a'].value * x + result.params['b'].value

            plt.figure(figsize=(8, 6))
            plt.scatter(x, original_OC, label='Original O-C data', color='blue')
            plt.plot(x, fitted_line, label='Fitted linear trend', color='red')
            plt.xlabel('E_corr')
            plt.ylabel('O-C')
            plt.title('Linear Fit to O-C Data')
            plt.legend()
            plt.show()

        return result


    def create_fit_model(self, result, first=None):
        """
        Construct a new OC_model instance with parameters set to fit result values.

        Parameters:
            result (lmfit.MinimizerResult): Fit output.
            first: Unused placeholder.
        Returns:
            OC_model: Model with updated parameters and uncertainties.
        """
        new_model = copy.deepcopy(self.model)
        fitted_params = result.params    

        for param_name, param in fitted_params.items():
            splitted_name = param_name.split("_")
            component_name = splitted_name[-1]
            
            for component in new_model.model_components:
                if component.name == component_name:
                    parameter = "_".join(splitted_name[:-1])
                    
                    # Update the parameter value
                    fitted_value = param.value
                    fitted_stderr = param.stderr if (param.stderr is not None and not np.isnan(param.stderr)) else np.nan

                    setattr(component, parameter, fitted_value)
                    getattr(component, parameter).std = fitted_stderr
                    component.params[parameter].std = fitted_stderr

                    # Unit check (optional, keep if needed)
                    current_param = getattr(component, parameter)
                    if current_param.unit != component._main_units[parameter]:
                        new_value = current_param.value
                        setattr(component, parameter, new_value)
        return new_model
        
    def create_model_from_samples(self, samples):
        """
        Generate model instance from median and std of MCMC samples for free parameters.

        Parameters:
            samples (np.ndarray): MCMC samples shape (n_samples, n_parameters).
        Returns:
            OC_model: New model with values from posterior medians.
        """
        variable_indices = []
        for comp in self.model.model_components:
            for par in comp.params.values():
                if par.vary:
                    variable_indices.append(len(variable_indices))

        medians = np.median(samples, axis=0)           # shape: (dim_count,)
        stds    = np.std(samples, axis=0, ddof=0)      # shape: (dim_count,)

        new_model = copy.deepcopy(self.model)
        flat_counter = 0  # single counter that walks through *all* parameters

        for comp in new_model.model_components:
            for name, par in comp.params.items():
                if par.vary:
                    par.value = medians[flat_counter]
                    par.std   = stds[flat_counter]
                    flat_counter += 1

        new_model.Ref_mintime = self.data.Ref_mintime
        new_model.Ref_period  = self.data.Ref_period

        return new_model
        
    def read_samples(self, sample_file):
        """
        Load MCMC samples saved in a text file.

        Parameters:
            sample_file (str): Path to samples file.
        Returns:
            np.ndarray: Array of loaded samples.
        """
        samples = np.loadtxt(sample_file)
        return samples

    def trace_plot(
            self,
            sampler,
            outfile_tag: str = "",
            save_plots: bool = False,
            show: bool = False,
            labels_override: list[str] | None = None,
            *,
            n_cols: int = 3,
            alpha: float = 0.7,
            fig_width_per_col: float = 4.0,
            fig_height_per_row: float = 2.2,
            dpi: int = 300,
    ) -> None:
        """
        Plot MCMC trace lines for each free parameter.

        Parameters:
            sampler (emcee.EnsembleSampler): Sample generator after run.
            outfile_tag (str): Suffix for plot filenames.
            save_plots (bool): Save plot images if True.
            show (bool): Show plot if True.
            labels_override (list[str]): Custom labels for parameters.
            n_cols (int): Number of subplot columns.
            alpha (float): Line opacity.
            fig_width_per_col (float): Width per column.
            fig_height_per_row (float): Height per row.
            dpi (int): Figure resolution.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        # ------------------------------------------------------------------ chain
        try:
            chain = sampler.get_chain()              # (n_steps, n_walkers, n_dim)  emcee ≥ 3
        except AttributeError:                       # emcee < 3
            chain = np.swapaxes(sampler.chain, 0, 1)  # (n_steps, n_walkers, n_dim)

        n_steps, n_walkers, n_dim = chain.shape

        # --------------------------------------------------------- build label list
        if labels_override is None:
            labels: list[str] = []
            for comp in self.model.model_components:
                for pname, par in comp.params.items():
                    if par.vary:
                        labels.append(f"{comp.name}_{pname}")
        else:
            if len(labels_override) != n_dim:
                raise ValueError(
                    f"labels_override has {len(labels_override)} items "
                    f"but the chain contains {n_dim} parameters."
                )
            labels = labels_override

        # ------------------------------------------------------------- make figure
        n_rows = int(np.ceil(n_dim / n_cols))
        fig_w  = n_cols * fig_width_per_col
        fig_h  = n_rows * fig_height_per_row

        plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

        for h in range(n_dim):
            ax = plt.subplot(n_rows, n_cols, h + 1)
            ax.plot(chain[:, :, h], alpha=alpha)
            ax.set_title(labels[h], fontsize=10)
            ax.set_xlabel("Step",  fontsize=8)
            ax.set_ylabel("Value", fontsize=8)
            ax.tick_params(axis="both", which="major", labelsize=8)

        plt.tight_layout()

        # -------------------------------------------------------------- save/show
        if save_plots:
            fname = f"{self.data.object_name}_trace_{outfile_tag}.png"
            plt.savefig(fname, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()

        plt.close()



    def corner_plot(self, samples, outfile_tag="", save_plots=False, show=False):
        """
        Generate a corner (pairwise) plot of MCMC samples with median markers.

        Parameters:
            samples (np.ndarray): Posterior samples.
            outfile_tag (str): Suffix for output file.
            save_plots (bool): Save the figure if True.
            show (bool): Display the figure if True.
        Returns:
            matplotlib.figure.Figure: Generated corner plot.
        """
        labels = []
        for model_component in self.model.model_components:
            for parameter in model_component.params.keys():
                if model_component.params[parameter].vary:
                    labels.append(model_component.name + "_" + parameter)

        medians = np.median(samples, axis=0)

        fig = corner.corner(
            samples, 
            labels=labels, 
            label_kwargs={"fontsize": 16},
            show_titles=True, 
            title_fmt=".2f",
            title_kwargs={"fontsize": 10},
            hist_kwargs={"linewidth": 1.5},
            plot_datapoints=True, 
            plot_contours=True, 
            plot_density=True,
            color="black"
        )

        axes = np.array(fig.axes).reshape((len(labels), len(labels)))

        # Add cross lines at the median values.
        for i in range(len(labels)):
            for j in range(i):
                ax = axes[i, j]
                ax.axvline(medians[j], color="black", linestyle="-", linewidth=1.5)
                ax.axhline(medians[i], color="black", linestyle="-", linewidth=1.5)
                ax.plot(medians[j], medians[i], "s", color="black", markersize=6)

        # Add vertical median lines on the diagonal histograms.
        for i in range(len(labels)):
            ax = axes[i, i]
            ax.axvline(medians[i], color="black", linestyle="-", linewidth=1.5)

        if show:
            plt.show()
        if save_plots:
            fig.savefig(self.data.object_name + "_corner_" + str(outfile_tag) + ".png")
        plt.close()
        return fig

    def clear_emcee_sample(self, samples, threshold=0, order=0, clear_count=np.inf, inplace=False):
        """
        Filter MCMC samples by removing outliers in histogram space.

        Parameters:
            samples (np.ndarray): Raw MCMC samples.
            threshold (float): Count ratio threshold.
            order (int): Direction of column iteration (0 or 1).
            clear_count (int): Max clearing iterations.
            inplace (bool): Update fitted_model if True.
        Returns:
            np.ndarray: Filtered MCMC samples.
        """
        samples2 = samples

        if order == 0:
            l = range(samples2.shape[1])
        elif order == 1:
            l = range(samples2.shape[1] - 1, -1, -1)

        for col_index in l:
            count = 0
            cc = clear_count
            while cc != 0:
                data_column = samples2[:, col_index]
                if data_column.size == 0:
                    break

                counts, bin_edges = np.histogram(data_column, bins=20)
                current_threshold = threshold * max(counts)
                start_index = np.argmax(counts)
                end_index = np.argmax(counts)
                for i in range(start_index):
                    if counts[start_index] <= current_threshold:
                        start_index += 1
                        break
                    else:
                        start_index -= 1
                else:
                    start_index = 0
                for i in range(len(counts) - end_index - 1):
                    if counts[end_index] <= current_threshold:
                        end_index -= 1
                        break
                    else:
                        end_index += 1
                else:
                    end_index = len(counts) - 1

                start_point = bin_edges[start_index]
                end_point = bin_edges[end_index + 1]
                if data_column.size == 0 or max(data_column) > end_point + (end_point - start_point) * 0.1 or min(data_column) < start_point - (end_point - start_point) * 0.1:
                    count += 1
                    samples2 = samples2[(data_column < end_point) & (data_column > start_point)]
                    if samples2.size == 0:
                        print("No more data to remove. Exiting loop.")
                        break
                    cc -= 1
                    continue
                else:
                    break
                
        if inplace:
            self.fitted_model = self.create_model_from_samples(samples2)

        return samples2
    
    def find_orbital_parameters(self,
                                inc: float, inc_std: float,
                                m1: float,  m1_std: float,
                                m2: float,  m2_std: float) -> "pd.DataFrame":
        """
        Compute orbital parameters for each model component (LiTE‐based or raw)
        and return them as a tidy pandas DataFrame with nominal values and uncertainties.

        Parameters:
            inc (float): Inclination angle in degrees.
            inc_std (float): Uncertainty of inclination.
            m1 (float): Primary mass.
            m1_std (float): Uncertainty of primary mass.
            m2 (float): Secondary mass.
            m2_std (float): Uncertainty of secondary mass.

        Returns:
            pd.DataFrame: MultiIndex (component, parameter) with columns
                        ['value', 'uncertainty'].
        """
        import numpy as np
        import pandas as pd
        from uncertainties import ufloat
        import orbit_param_calculator

        # conversion constants
        DAY2SEC   = 86400.0
        JUPITER_M = 1047.56

        # wrap global inputs as ufloats
        inc_uf = ufloat(np.deg2rad(inc), np.deg2rad(inc_std))
        m1_uf  = ufloat(m1, m1_std)
        m2_uf  = ufloat(m2, m2_std)
        RefP   = ufloat(self.data.Ref_period, 0.0)

        raw_results = {}

        for comp in self.fitted_model.model_components:
            name = comp.name

            if hasattr(comp, "amp"):
                # — LiTE‐based calculation path —

                # 1) convert units for P_LiTE, amp, omega, T_LiTE
                print("tts=", comp.T_LiTE.std)
                P_val = _unit_conv(comp.P_LiTE.value, comp.P_LiTE.unit,
                                comp._main_units["P_LiTE"],
                                ref_period=self.data.Ref_period,
                                ref_mintime=self.data.Ref_mintime,
                                parameter_name="P_LiTE",
                                JD_convertable=comp.P_LiTE.JD_convertable)
                P_std = _unit_conv(comp.P_LiTE.std,  comp.P_LiTE.unit,
                                comp._main_units["P_LiTE"],
                                ref_period=self.data.Ref_period,
                                ref_mintime=self.data.Ref_mintime,
                                parameter_name="P_LiTE",
                                JD_convertable=comp.P_LiTE.JD_convertable)

                amp_val = _unit_conv(comp.amp.value, comp.amp.unit,
                                    comp._main_units["amp"],
                                    ref_period=self.data.Ref_period,
                                    ref_mintime=self.data.Ref_mintime,
                                    parameter_name="amp",
                                    JD_convertable=comp.amp.JD_convertable)
                amp_std = _unit_conv(comp.amp.std,  comp.amp.unit,
                                    comp._main_units["amp"],
                                    ref_period=self.data.Ref_period,
                                    ref_mintime=self.data.Ref_mintime,
                                    parameter_name="amp",
                                    JD_convertable=comp.amp.JD_convertable)

                omega_val = _unit_conv(comp.omega.value, comp.omega.unit,
                                    comp._main_units["omega"],
                                    ref_period=self.data.Ref_period,
                                    ref_mintime=self.data.Ref_mintime,
                                    parameter_name="omega",
                                    JD_convertable=comp.omega.JD_convertable)
                omega_std = _unit_conv(comp.omega.std,  comp.omega.unit,
                                    comp._main_units["omega"],
                                    ref_period=self.data.Ref_period,
                                    ref_mintime=self.data.Ref_mintime,
                                    parameter_name="omega",
                                    JD_convertable=comp.omega.JD_convertable)

                T_val = _unit_conv(comp.T_LiTE.value, comp.T_LiTE.unit,
                                comp._main_units["T_LiTE"],
                                ref_period=self.data.Ref_period,
                                ref_mintime=self.data.Ref_mintime,
                                parameter_name="T_LiTE",
                                JD_convertable=comp.T_LiTE.JD_convertable)
                if comp.T_LiTE.unit == "BJD":
                    tlite_temp_unit = "day"
                else:
                    tlite_temp_unit = comp.T_LiTE.unit
                T_std = _unit_conv(comp.T_LiTE.std,  tlite_temp_unit,
                                comp._main_units["T_LiTE"],
                                ref_period=self.data.Ref_period,
                                ref_mintime=self.data.Ref_mintime,
                                parameter_name="T_LiTE",
                                JD_convertable=comp.T_LiTE.JD_convertable)

                ecc_uf = ufloat(comp.e.value, comp.e.std)

                print("tts=", T_std)

                # 2) wrap in ufloats
                P_ratio  = ufloat(P_val,   P_std)
                amp_days = ufloat(amp_val, amp_std)
                omega_uf = ufloat(np.deg2rad(omega_val), np.deg2rad(omega_std))
                T_uf     = ufloat(T_val,   T_std)
                T_BJD    = T_uf * self.data.Ref_period + self.data.Ref_mintime

                # 3) period conversions
                P_days = P_ratio * RefP
                P_yr   = orbit_param_calculator.period_in_years(P_ratio, RefP)

                # 4) semi-major axes
                a12sini_au = orbit_param_calculator.a12sini(amp_days, ecc_uf, omega_uf)
                a12_au     = orbit_param_calculator.a12(a12sini_au, inc_uf)

                # 5) mass function & tertiary mass
                f_mass      = orbit_param_calculator.mass_func(P_yr, a12sini_au)
                m3sini3     = orbit_param_calculator.m3sini3(f_mass, m1_uf, m2_uf)
                m3sini3_jup = m3sini3 * JUPITER_M

                # 6) outer semi-major axis projection
                a3sini3 = orbit_param_calculator.a3sini3(a12_au, m1_uf, m2_uf, m3sini3)

                # 7) collect results in a dict
                raw_results[name] = {
                    "P_days"       : P_days,
                    "P_years"      : P_yr,
                    "amp_days"     : amp_days,
                    "amp_seconds"  : amp_days * DAY2SEC,
                    "eccentricity" : ecc_uf,
                    "omega_deg"    : ufloat(omega_val, omega_std),
                    "a12sini_AU"   : a12sini_au,
                    "a12_AU"       : a12_au,
                    "f_mass"       : f_mass,
                    "m3sini_msol"  : m3sini3,
                    "m3sini_mjup"  : m3sini3_jup,
                    "a3sini_AU"    : a3sini3,
                    "T_LiTE"       : T_BJD
                }

            else:
                # — No amp: just dump raw parameters —
                raw_results[name] = {
                    pname: (param.value if hasattr(param, "value") else param)
                    for pname, param in comp.params.items()
                }

        # Flatten into a pandas DataFrame
        rows = []
        for comp_name, params in raw_results.items():
            for pname, pval in params.items():
                if hasattr(pval, "nominal_value") and hasattr(pval, "std_dev"):
                    nominal = pval.nominal_value
                    uncert  = pval.std_dev
                else:
                    nominal = pval
                    uncert  = None
                rows.append({
                    "component":   comp_name,
                    "parameter":   pname,
                    "value":       nominal,
                    "uncertainty": uncert
                })

        df = pd.DataFrame(rows)
        df.set_index(["component", "parameter"], inplace=True)
        return df


    def log_likelihood(self, variable_params, m1, m2, inc):
        """
        Compute the log-likelihood of the data given model parameters.

        Parameters:
            variable_params (array-like): Free parameters for model.
            m1 (float): Primary mass.
            m2 (float): Secondary mass.
            inc (float): Inclination angle.

        Returns:
            float: Log-likelihood value.
        """
        observed_oc = self.data.OC
        observed_errors = self.data.Errors
        simulated_oc = self.total_oc_delay(variable_params, m1, m2, inc, Ecorr=None, mintimes_in_data=True)
        residuals = observed_oc - simulated_oc
        return -0.5 * np.sum(((residuals)**2 / observed_errors**2) + np.log(observed_errors**2))

    def log_prior(self, variable_params):
        """
        Evaluate the log-prior probability of the parameters.

        Parameters:
            variable_params (array-like): Free parameters for model.

        Returns:
            float: Log-prior (0 if uniform within bounds, -inf otherwise).
        """
        ln_sum = 0.0
        param_index = 0
        for component in self.model.model_components:
            for param_name, param_obj in component.params.items():
                if param_obj.vary:
                    param_value = variable_params[param_index]
                    if (param_obj.min is not None and param_value < param_obj.min) or (param_obj.max is not None and param_value > param_obj.max):
                        return -np.inf
                    if self.prob_prior:
                        ln_sum += fit.gaussian(param_value, param_obj.value, param_obj.std / 2.0)
                    param_index += 1
        return ln_sum

    def log_prob(self, variable_params, m1, m2, inc):
        """
        Compute the log-posterior as sum of log_prior and log_likelihood.

        Parameters:
            variable_params (array-like): Free parameters for model.
            m1 (float): Primary mass.
            m2 (float): Secondary mass.
            inc (float): Inclination angle.

        Returns:
            float: Log-posterior probability.
        """
        lp = self.log_prior(variable_params)
        if not np.isfinite(lp):
            return -np.inf
        likelihood = self.log_likelihood(variable_params, m1, m2, inc)
        return lp + likelihood if np.isfinite(likelihood) else -np.inf      

    def total_oc_delay(self, variable_params, m1, m2, inc, Ecorr=None, fix_units_first=False, mintimes_in_data=False):
        """
        Calculate total O-C delay contributions from all model components.

        Parameters:
            variable_params (array-like): List of free parameters.
            m1 (float): Primary mass.
            m2 (float): Secondary mass.
            inc (float): Inclination angle.
            Ecorr (np.ndarray, optional): Epoch corrections.
            Mintimes (np.ndarray, optional): Observation times of minima.
            fix_units_first (bool): If True, apply unit conversion first.

        Returns:
            np.ndarray: Total O-C delay values.
        """
        if fix_units_first:
            model = self.model.fix_units()
            converted_variable_params = []
            param_index = 0
            for component in model.model_components:
                for param_name, param_obj in component.params.items():
                    if param_obj.vary:
                        # Convert the variable_params[param_index] to the fixed unit
                        converted_value = _unit_conv(
                            variable_params[param_index],
                            from_unit=getattr(component, param_name).unit,  # original unit
                            to_unit=component._main_units[param_name],     # fixed unit
                            ref_period=self.data.Ref_period,
                            ref_mintime=self.data.Ref_mintime,
                            parameter_name=param_name,
                            JD_convertable=getattr(component, param_name).JD_convertable
                        )
                        converted_variable_params.append(converted_value)
                        param_index += 1
            variable_params = converted_variable_params
        else:
            model = self.model

        Ecorr = Ecorr if Ecorr is not None else self.data.Ecorr
        simulated_oc = 0
        param_index = 0
        lite_params = []
        lin_exists = False
        for component in model.model_components:
            if isinstance(component, LiTE_abspar):
                lite_param_set = {}
                for param_name, param_obj in component.params.items():
                    if param_obj.vary:
                        param_obj.value = variable_params[param_index]
                        lite_param_set[param_name] = param_obj.value
                        param_index += 1
                    else:
                        lite_param_set[param_name] = param_obj.value
                lite_params.append(lite_param_set)
            else:
                params, param_index = self._extract_component_params(component, variable_params, param_index)
                if isinstance(component, Lin):
                    dP, dT = params
                    lin_exists = True
                simulated_oc += component.individual_model(Ecorr, *params)

        if not lin_exists:
            dP = dT = 0
        if lite_params:
            if mintimes_in_data:
                Mintimes = self.data.Mintimes
            else:
                Mintimes = Ecorr * (self.data.Ref_period + dP) + (self.data.Ref_mintime + dT)
            simulated_oc += self.simulate_oc_delay_lite(m1, m2, inc, lite_params, Mintimes)
        return simulated_oc

    def _extract_component_params(self, component, variable_params, start_index):
        """
        Internal: Extract a slice of variable_params for a specific component.

        Parameters:
            component (object): Model component instance.
            variable_params (array-like): Flat list of all free parameters.
            start_index (int): Starting index in variable_params for this component.

        Returns:
            tuple: (params_list, next_index) updated index after extraction.
        """
        params = []
        for param_name, param_obj in component.params.items():
            param_value = variable_params[start_index] if param_obj.vary else param_obj.value
            params.append(param_value)
            if param_obj.vary:
                start_index += 1
        return params, start_index

    def simulate_oc_delay_lite(self, m1, m2, system_inclination_deg, lite_params, mintimes):
        """
        Compute OC delays using analytical LiTE formula for a single component.

        Parameters:
            m1 (float): Primary mass.
            m2 (float): Secondary mass.
            system_inclination_deg (float): System inclination in degrees.
            lite_params (dict): Parameters for LiTE effect.
            mintimes (np.ndarray): Observation times of minima.

        Returns:
            np.ndarray: LiTE-based O-C delays.
        """
        # Extract parameters, ensuring lite_params is a list of dictionaries
        times_of_p = [p.get('T_LiTE', Parameter(0)) for p in lite_params]
        periods = [p.get('P_LiTE', Parameter(0)) for p in lite_params]
        eccs = [p.get('ecc', Parameter(0)) for p in lite_params]
        omegas = [p.get('omega', Parameter(0)) for p in lite_params]
        masses = [p.get('mass', Parameter(0)) for p in lite_params]
        incs = [p.get('inc', Parameter(0)) for p in lite_params]
        return simulate_oc_delay(m1, m2, system_inclination_deg, times_of_p, periods, eccs, omegas, masses, incs, mintimes, integrator=self.integrator)

    @staticmethod
    def gaussian(p, mu, sigma):
        """
        Compute Gaussian probability density function for value p.

        Parameters:
            p (float or array-like): Data value(s).
            mu (float): Mean of Gaussian.
            sigma (float): Standard deviation of Gaussian.

        Returns:
            float or np.ndarray: Probability density.
        """
        res = np.log(1.0 / (np.sqrt(2 * np.pi) * sigma)) - 0.5 * (p - mu)**2 / sigma**2
        return res


    def fit_model_prob(self, walker=20, steps=300, burn_in=0, threads=4, std_scale=0.05,
                       create_sample_file=False, trace_plot=True,
                       corner_plot=True, fit_plot=True, save_plots=False, show_plots=True,
                       prob_prior=True, return_samples=True, return_sampler=False,
                       multiprocessing=True, integrator="IAS15", ias15_accuracy=1e-8):
        """
        Run MCMC sampling with emcee to approximate the posterior distribution of model parameters.

        Parameters:
            walker (int): Number of walkers (chains) in the ensemble.
            steps (int): Total number of steps (iterations) to run per walker.
            burn_in (int): Number of initial samples to discard from each chain as burn-in.
            threads (int): Number of parallel threads to use for sampling.
            std_scale (float): Fractional scale for dispersing initial walker positions.
            create_sample_file (bool): If True, write the post-burn-in samples to a text file.
            trace_plot (bool): If True, generate and optionally save/display the trace diagnostic plot.
            corner_plot (bool): If True, generate and optionally save/display the corner plot.
            fit_plot (bool): If True, plot the sampled posterior distributions over the data.
            save_plots (bool): If True, save any generated plots to disk.
            show_plots (bool): If True, display plots interactively.
            prob_prior (bool): If True, include prior probability in the log-posterior calculation.
            return_samples (bool): If True, return the flattened sample array.
            return_sampler (bool): If True, return the emcee sampler instance as well.
            multiprocessing (bool): If True, use multiprocessing Pool for parallel sampling.
            integrator (str or None): Integrator name to pass to simulate_oc_delay.

        Returns:
            np.ndarray or tuple: By default returns the flattened samples array;
            if return_sampler is True, returns (samples, sampler).
        """
        import copy
        import os
        import numpy as np
        import emcee
        from multiprocessing import Pool
        import matplotlib.pyplot as plt

        # Create a deepcopy of this fitter and fix its model units
        fit2 = copy.deepcopy(self)
        fit2.model.Ref_period = fit2.data.Ref_period
        fit2.model.Ref_mintime = fit2.data.Ref_mintime
        fit2.model = fit2.model.fix_units()

        fit2.integrator = integrator
        fit2.ias15_accuracy = ias15_accuracy

        if burn_in > steps:
            raise ValueError("Burn-in value cannot be greater than the number of steps.")

        # Check for incompatible model components
        lite_abspar_count = sum(isinstance(c, LiTE_abspar) for c in fit2.model.model_components)
        lite_count        = sum(isinstance(c, LiTE)        for c in fit2.model.model_components)
        if lite_abspar_count > 0 and lite_count > 0:
            raise ValueError("Cannot have both LiTE_abspar and LiTE components in the model.")
        if lite_abspar_count > 0:
            if fit2.data.m1 is None or fit2.data.m2 is None:
                raise ValueError("m1, m2, and inc must be provided for LiTE_abspar model.")
            m1, m2, inc = fit2.data.m1, fit2.data.m2, fit2.data.inc
        else:
            m1, m2, inc = 0, 0, 0

        fit2.prob_prior = prob_prior

        # Initialize walker positions and dimensionality
        initial_positions, nwalkers, ndim = fit2._initialize_sampling_params(walker, std_scale)

        # Run the MCMC sampler
        if multiprocessing:
            with Pool(threads) as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, fit2.log_prob, args=(m1, m2, inc), pool=pool)
                sampler.run_mcmc(initial_positions, steps, progress=True)
        else:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, fit2.log_prob, args=(m1, m2, inc), threads=threads)
            sampler.run_mcmc(initial_positions, steps, progress=True)

        # Retrieve the full chain: shape = (nsteps, nwalkers, ndim)
        chains = sampler.get_chain(flat=False)
        print(f"Using all {chains.shape[1]} walkers without filtering for stuck walkers.")

        # Build a list of unique parameter keys: "ComponentName.param_name"
        variable_keys = []
        for comp in fit2.model.model_components:
            for pname, pobj in comp.params.items():
                if pobj.vary:
                    key = f"{comp.name}.{pname}"
                    variable_keys.append(key)

        # Map each key to its index in the chain's last axis
        param_indices = {key: idx for idx, key in enumerate(variable_keys)}

        # Convert the chain into user-specified units
        chains_conv = chains.copy()
        for comp in fit2.model.model_components:
            for pname, pobj in comp.params.items():
                if not pobj.vary:
                    continue
                key = f"{comp.name}.{pname}"
                idx = param_indices[key]
                # convert from internal main_units to user unit
                chains_conv[:, :, idx] = _unit_conv(
                    chains_conv[:, :, idx],
                    comp._main_units[pname],
                    pobj.unit,
                    ref_period=fit2.data.Ref_period,
                    ref_mintime=fit2.data.Ref_mintime,
                    parameter_name=pname,
                    JD_convertable=pobj.JD_convertable
                )

        # Flatten and apply burn-in
        samples_full = chains_conv.reshape(-1, ndim)
        samples      = chains_conv[burn_in:, :, :].reshape(-1, ndim)
        if samples.size == 0:
            raise ValueError("No samples remain after applying burn_in.")

        # Create a fitted model from the posterior samples
        self.fitted_model = self.create_model_from_samples(samples)

        # Prepare an output filename tag
        identifier  = "".join([c.name for c in fit2.model.model_components])
        outfile_tag = f"{nwalkers}_{steps}_{identifier}"
        base_fn     = f"{fit2.data.object_name}_prios_{outfile_tag}.out"
        filename    = base_fn
        counter     = 1
        while os.path.exists(filename):
            filename = f"{base_fn}_{counter}.out"
            counter += 1
        outfile_tag = f"{outfile_tag}_{counter}"

        # Optionally save samples to file
        if create_sample_file:
            sample_fn = f"{fit2.data.object_name}_emcee_samples_{outfile_tag}.out"
            print(f"Saving MCMC samples to {sample_fn}...")
            np.savetxt(sample_fn, samples, delimiter=" ", fmt="%.8e")

        # Generate diagnostic plots
        if trace_plot:
            self.trace_plot(sampler, outfile_tag="filtered", save_plots=save_plots, show=show_plots)
        if corner_plot:
            self.corner_plot(samples,    outfile_tag="filtered", save_plots=save_plots, show=show_plots)
        if fit_plot:
            self.plot(samples=samples, show=show_plots)

        plt.show()

        # Return according to flags
        if return_sampler and not return_samples:
            return sampler
        if return_samples and not return_sampler:
            return samples
        return samples, sampler


    def _initialize_sampling_params(self, walker, std_scale):
        """
        Internal: Prepare initial sampler positions and labels for MCMC.

        Parameters:
            walker (int): Number of walkers.
            std_scale (float): Scale factor for dispersing initial positions.

        Returns:
            tuple: (initial_positions, walker_count, dim_count)
        """
        # Gather parameter centres, scaled sigmas, and hard bounds
        centers, scales, min_bounds, max_bounds = [], [], [], []
        for component in self.model.model_components:
            for param_obj in component.params.values():
                if param_obj.vary:
                    centers.append(param_obj.value)
                    # fallback to a small value if std is None
                    scales.append((param_obj.std or 1e-4) * std_scale)
                    min_bounds.append(param_obj.min if param_obj.min is not None else -np.inf)
                    max_bounds.append(param_obj.max if param_obj.max is not None else np.inf)

        walker_count = walker
        dim_count = len(centers)

        # Compute the uniform sampling window for each dimension
        centers = np.array(centers)
        scales = np.array(scales)
        min_bounds = np.array(min_bounds)
        max_bounds = np.array(max_bounds)

        # Low end = max(hard_min, center - scale), high end = min(hard_max, center + scale)
        low = np.maximum(min_bounds, centers - scales)
        high = np.minimum(max_bounds, centers + scales)

        # Draw uniformly in [low, high] for each walker and dimension
        initial_positions = np.random.uniform(
            low=low, 
            high=high, 
            size=(walker_count, dim_count)
        )

        return initial_positions, walker_count, dim_count
            
    def plot_orbit_gif(self,
                    output_file="orbit.gif",
                    time_count=1000,
                    real_inc=False,
                    epochs=None,
                    speed=1,
                    show_oc=True,
                    show_data=True):
        """
        Generate and save an animated GIF visualizing orbital motion with persistent trails,
        at adjustable playback speed, and optional O–C subplot and observed-data overlay.

        Parameters:
            output_file (str): Filename for saved GIF.
            time_count (int): Number of frames/time steps.
            real_inc (bool): Use real inclination if True.
            epochs (array-like): Optional manual epochs array.
            speed (float): Playback speed multiplier (1 = normal, 2 = 2× faster, etc.).
            show_oc (bool): If True, include the O–C subplot; otherwise omit it.
            show_data (bool): If True and show_oc=True, plot the observed O–C data points.
        """
        import copy
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        if speed <= 0:
            raise ValueError("`speed` must be > 0")

        # --- prepare a clean copy of the fit ---
        fit2 = copy.deepcopy(self)
        fit2.model.Ref_period  = fit2.data.Ref_period
        fit2.model.Ref_mintime = fit2.data.Ref_mintime

        # Unwrap masses & inclination
        m1 = getattr(fit2.data.m1,  "value", fit2.data.m1)
        m2 = getattr(fit2.data.m2,  "value", fit2.data.m2)
        system_inc = getattr(fit2.data.inc, "value", fit2.data.inc)

        # Extract LiTE component parameters
        times_of_p, periods, eccs, omegas, masses, incs_used = [], [], [], [], [], []
        for comp in fit2.model.model_components:
            if isinstance(comp, LiTE_abspar):
                p = comp.params
                times_of_p.append(p["T_LiTE"].value)
                periods.append(    p["P_LiTE"].value)
                eccs.append(       p["ecc"].value)
                omegas.append(     p["omega"].value)
                masses.append(     p["mass"].value)
                incs_used.append(  p["inc"].value if real_inc else 0.0)

        # Build smooth mintimes
        orig_mt = fit2.data.Mintimes
        mintimes = np.linspace(orig_mt.min(), orig_mt.max(), time_count)

        # Reset fit2 for O–C
        fit2 = copy.deepcopy(self)
        fit2.data.df = pd.DataFrame()
        fit2.data.Mintimes = mintimes
        if epochs is None:
            fit2.data.Ecorr = np.linspace(
                fit2.data.Ecorr.min(), fit2.data.Ecorr.max(), time_count
            )
        else:
            fit2.data.Ecorr = np.array(epochs)
        fit2.model.epochs = fit2.data.Ecorr

        # Freeze all parameters
        for mc in fit2.model.model_components:
            for param in mc.params.values():
                param.vary = False

        # Compute model O–C delay
        total_oc_delay = fit2.total_oc_delay([], m1, m2,
                                            system_inc,
                                            Ecorr=fit2.data.Ecorr)

        # Unwrap linear terms
        dT = dP = 0.0
        for mc in fit2.model.model_components:
            if isinstance(mc, Lin):
                dT = getattr(mc.dT, "value", mc.dT)
                dP = getattr(mc.dP, "value", mc.dP)
        mintimes = (fit2.data.Ecorr * (self.data.Ref_period + dP)
                + (self.data.Ref_mintime + dT))

        # Simulate positions
        _, positions = simulate_oc_delay(
            m1=m1, m2=m2,
            system_inclination_deg=(system_inc if real_inc else 0.0),
            times_of_p=times_of_p,
            periods=periods, eccentricities=eccs, omegas_deg=omegas,
            masses=masses, incs=incs_used,
            mintimes=mintimes,
            return_pos=True, integrator=self.integrator
        )
        for key in positions:
            positions[key] = np.array(positions[key])

        # Set up figure & axes
        if show_oc:
            fig, (ax_orbit, ax_oc) = plt.subplots(2, 1, figsize=(6, 10))
            fig.subplots_adjust(hspace=0.4)
        else:
            fig, ax_orbit = plt.subplots(1, 1, figsize=(6, 6))
            ax_oc = None

        # Orbit plot limits
        max_r = max(np.linalg.norm(positions[i], axis=1).max() for i in positions)
        ax_orbit.set_xlim(-max_r-1, max_r+1)
        ax_orbit.set_ylim(-max_r-1, max_r+1)
        ax_orbit.set_xlabel("X (AU)")
        ax_orbit.set_ylabel("Y (AU)")
        ax_orbit.grid(True)

        # Draw observer’s line-of-sight arrow when not using real inclination
        if not real_inc:
            arrow_props = dict(arrowstyle='->', lw=1.5, color='k')
            # Arrow from just beyond the top edge pointing inward
            ax_orbit.annotate(
                '', 
                xy=(0,  max_r * 0.8), 
                xytext=(0,  max_r), 
                arrowprops=arrow_props
            )
            ax_orbit.text(
                0, max_r * 1.05, 
                'Observer', 
                ha='center', 
                va='bottom', 
                fontsize=10
            )

        # Create markers & trails, with first labeled "Binary Star"
        lines_markers, lines_trails = [], []
        for idx, key in enumerate(sorted(positions)):
            label = "Binary Star" if idx == 0 else f"Body {idx}"
            mline, = ax_orbit.plot([], [], 'o', label=label)
            tline, = ax_orbit.plot([], [], '-', alpha=0.4)
            lines_markers.append(mline)
            lines_trails.append(tline)
        ax_orbit.legend()

        # O–C subplot setup…
        if show_oc:
            ax_oc.set_xlim(fit2.data.Ecorr.min(), fit2.data.Ecorr.max())
            ax_oc.set_xlabel("Epoch")
            ax_oc.set_ylabel("Total O–C Delay (Days)")
            ax_oc.grid(True)
            if show_data:
                ax_oc.plot(self.data.Ecorr, fit2.data.OC, 'r.', label="Observed O–C")
            else:
                ymn, ymx = total_oc_delay.min(), total_oc_delay.max()
                ax_oc.set_ylim(ymn, ymx)
            line_total_oc, = ax_oc.plot([], [], '--', color='k', label="Model O–C")
            ax_oc.legend()

        # Animation update function…
        def update(frame):
            artists = []
            for marker, trail in zip(lines_markers, lines_trails):
                idx = lines_markers.index(marker)
                x, y = positions[idx][frame]
                marker.set_data([x], [y])
                xs = positions[idx][:frame+1, 0]
                ys = positions[idx][:frame+1, 1]
                trail.set_data(xs, ys)
                artists.extend([marker, trail])
            if show_oc:
                line_total_oc.set_data(
                    fit2.data.Ecorr[:frame+1],
                    total_oc_delay[:frame+1]
                )
                artists.append(line_total_oc)
            return artists

        # Save animation
        base_interval = 100
        interval = max(1, int(base_interval / speed))
        ani = animation.FuncAnimation(
            fig, update, frames=time_count, interval=interval, blit=True
        )
        ani.save(output_file, writer="pillow")
        plt.close(fig)
        print(f"Saved {output_file} at {speed}× speed, "
            f"{'with' if show_oc else 'no'} O–C, "
            f"{'showing' if show_data else 'hiding'} data.")

    def calculate_uncertanities(self, samples, epochs, mintimes,
                                q_lower=16, q_upper=84):
        """
        Calculate ±1σ uncertainty envelopes from MCMC samples.

        Parameters:
            samples (np.ndarray): Flattened walker samples; shape (N, n_free) for variable parameters.
            epochs (array-like): Epoch numbers at which to evaluate the O–C curve.
            mintimes (array-like): Corresponding observation times for each epoch.
            q_lower (float): Lower percentile (default: 16).
            q_upper (float): Upper percentile (default: 84).

        Returns:
            tuple: (sigma_plus, sigma_minus) envelopes relative to the median O–C curve.
        """
        import numpy as np

        # --- make sure epochs/mintimes are arrays ---
        epochs   = np.asarray(epochs)
        mintimes = np.asarray(mintimes)

        # --- collect exactly the variable parameters in the same order MCMC used ---
        var_params = []
        for comp in self.model.model_components:
            for name, p in comp.params.items():
                if p.vary:
                    var_params.append((comp, name))

        n_free = len(var_params)
        if samples.shape[1] != n_free:
            raise ValueError(
                f"Expected samples.shape[1]={n_free} (number of p.vary=True), "
                f"but got {samples.shape[1]}"
            )

        # --- build one O–C curve per sample, skipping any NaN runs ---
        curves = []
        for theta in samples:
            # pass only the sampled values in order
            curve = self.total_oc_delay(
                list(theta),
                self.data.m1, self.data.m2,
                self.data.inc, epochs,
                fix_units_first=True
            )
            # coerce to 1-D array of length len(epochs)
            curve = np.atleast_1d(curve)
            if curve.shape != epochs.shape:
                curve = np.full_like(epochs, curve.flat[0])
            # drop any that went to nan
            if np.isnan(curve).any():
                continue
            curves.append(curve)

        if not curves:
            raise RuntimeError("No valid O–C curves: every sample produced NaNs.")

        # --- stack & compute percentiles ---
        curves = np.vstack(curves)  # shape = (n_valid_samples, n_epochs)
        y_lo  = np.nanpercentile(curves, q_lower, axis=0)
        y_hi  = np.nanpercentile(curves, q_upper, axis=0)
        y_med = np.nanpercentile(curves, 50,     axis=0)

        # --- return +1σ & –1σ relative to the median ---
        sigma_plus  = y_hi  - y_med
        sigma_minus = y_med - y_lo
        return sigma_plus, sigma_minus

    def plot(self, **kwargs):
        """
        Plot the O–C data and uncertainty bands from MCMC samples.

        Parameters:
            samples (np.ndarray): Posterior samples for plotting uncertainty regions.
            **kwargs: Additional keyword arguments passed to the plotting function (fit_plot),
                      such as figsize, title, labels, and colors.
        """
        model = self.fitted_model if hasattr(self, "fitted_model") else self.model
        return fit_plot(
            model=model,
            data=self.data,
            integrator=self.integrator,
            **kwargs
        )



    
class Parameter:  
    """
    Represents a single model parameter with its value, unit, bounds, and metadata.

    Attributes:
        value (float): Numerical value of the parameter.
        unit (str or None): Physical unit label (e.g., 'day', 'deg').
        min (float or None): Lower bound for fitting.
        max (float or None): Upper bound for fitting.
        vary (bool): If True, parameter is free during fitting.
        std (float): Estimated uncertainty of the parameter.
    """
    def __init__(self, value, unit=None, min=None, max=None, vary=True, JD_convertable=False, expr=None, brute_step=None, user_data=None, std=0):
        """
        Initialize a Parameter instance.

        Parameters:
            value (float): Initial parameter value.
            unit (str, optional): Unit label.
            min (float, optional): Minimum fitting bound.
            max (float, optional): Maximum fitting bound.
            JD_convertable (bool): If True, apply JD conversion logic.
            vary (bool): Whether the parameter is varied in fitting.
            expr (str, optional): Expression linking this parameter to others.
            brute_step (float, optional): Step size for brute-force scanning.
            user_data (any, optional): Auxiliary data for custom use.
            std (float): Initial uncertainty estimate.
        """
        self.value = value
        self.unit = unit
        self.JD_convertable = JD_convertable
        self.min = min
        self.max = max
        self.vary = vary
        self.brute_step = brute_step
        self.expr = expr
        self.user_data = user_data
        self.std = std
    
    def __repr__(self) -> str:
        """
        Returns a string representation of the Parameter object.

        Returns:
            str: String representation of the Parameter object.
        """
        return f"{self.value} {self.unit}"

class model_component:
    """
    Parent class for all model components (e.g., Lin, Quad, LiTE, etc.).

    Manages parameter assignment (wrapping raw floats into `Parameter` objects),
    enforces unit consistency, and provides generic plotting and summary utilities.

    Key Responsibilities:
      - Intercept attribute setting for any name in `_main_units` to keep
        the `params` dict in sync.
      - Provide `set_params()` to bulk–update parameters.
      - Offer a default `plot()` and `summary()` for inspection.
    """
    def __setattr__(self, __name, __value) -> None:
        if hasattr(self, "_main_units") and __name in self._main_units.keys():
            val = self._set_single_param(__name, __value)
            super().__setattr__(__name, val)
            self.params[__name] = val  # Ensure deep copy here
        elif __name == "params":
            super().__setattr__(__name, __value)  # Deep copy the params dictionary
            for key, v in __value.items():
                if key in self._main_units:   
                    setattr(self, key, v)
            super().__setattr__(__name, __value)
        else:
            super().__setattr__(__name, __value)

    def set_params(self, params):
        """
        Bulk–assign multiple parameters from a dict.

        Each key→value pair is passed through `_set_single_param()`. Any
        parameter whose name starts with "T_" is flagged as `JD_convertable`.

        Args:
            params (dict): {param_name: float or Parameter}
        """
        for variable, value in params.items():
            setattr(self, variable, self._set_single_param(variable, value))
            if variable.startswith("T_"):
                getattr(self, variable).JD_convertable = True

    def _set_single_param(self, variable, value):
        """
        Wrap or update a single parameter into a `Parameter` object.

        - If `value` is float/int: update existing `Parameter.value` if present,
          else create `Parameter(value, unit)`.
        - If `value` is already a `Parameter`: enforce unit consistency and
          preserve any existing `JD_convertable` flag.
        - Otherwise: raise.

        Args:
            variable (str): parameter name
            value (float or Parameter)

        Returns:
            Parameter
        """
        if isinstance(value, (int, float)):
            if hasattr(self, variable) and isinstance(getattr(self, variable), Parameter):
                getattr(self, variable).value = value
                return getattr(self, variable)
            else:
                return Parameter(value, self._main_units[variable])
        elif isinstance(value, Parameter):
            ### !!!hardcode to stay jd convertable!!!
            if hasattr(self, variable):
                value.JD_convertable = getattr(self, variable).JD_convertable
            if value.unit is None:
                value.unit = self._main_units[variable]
            if variable == "T_LiTE":
                value.JD_convertable = True
            return value
        else:
            raise ValueError(f"{variable} must be a float or Parameter")
        
    def plot(self, epochs=None, show=True):
        """
        Quick-plot this component’s `individual_model` vs. `epochs`.

        Args:
            epochs (array-like, optional): X-axis values; defaults to
                                            `self.epochs` if present.
            show (bool): If True, calls `plt.show()`.
        """
        if epochs is None:
            epochs = self.epochs
        plt.plot(epochs, self.individual_model(epochs))
        plt.xlabel("Epoch")
        plt.ylabel("OC (d)")
        if show:
            plt.show()

    def __repr__(self) -> str:
        """
        Provides a string representation of the model_component instance, primarily its name.

        Returns:
            str: String representation of the model_component instance.
        """
        return self.name
    
    def summary(self, show=False, return_type = "dict"):
        """
        Returns the parameters and name of the model_component instance as a dictionary or DataFrame.

        Args:
            return_type (str): Type of summary to return. Options: "dict" or "dataframe" (default: "dict").

        Returns:
            dict or pandas.DataFrame: Summary of the model_component instance's parameters.
        """
        sum = {}
        sum["name"] = self.name
        for key, value in self.params.items():
            sum[key] = value
        if return_type == "dict":
            return sum
        if return_type == "dataframe":
            df = pd.DataFrame(list(sum.items()), columns=['Attributes', 'Values'])
            df.set_index('Attributes', inplace=True)            
            return df

class Lin(model_component):
    """
    Linear O–C trend: ΔP·epoch + ΔT offset.

    Parameters:
      - dP (slope, days per cycle)
      - dT (zero–point offset in days)
    """
    def __init__(self, params = {}, name="Lin"):
        """
        Initializes a new instance of the Lin class with specified temperature and pressure differences, and a name.

        Parameters:
        - params (dict): Dictionary containing parameters dT and dP.
        - name (str): The name of the instance. Default is "Lin".
        """
        self._main_units = {"dP": "day", "dT":"day"}
        self.params = {"dP":0, "dT":0} if params == {} else params
        self.name = name
        self.Ref_period = None
        self.set_params(params=params)

    def individual_model(self, x, dP=None, dT=None):
        """
        Calculate the linear O–C trend: ΔP·x + ΔT.

        Parameters
        ----------
        x : float or array_like
            Cycle (epoch) value(s) at which to evaluate the model.
        dP : float, optional
            Slope (ΔP) in days per cycle. If None, uses this component’s stored `dP`.
        dT : float, optional
            Offset (ΔT) in days. If None, uses this component’s stored `dT`.

        Returns
        -------
        float or ndarray
            Model prediction(s): `dP * x + dT`.
        """
        parameter = {}
        # print("dT=", dT)
        # print("dP=", dP)
        if dT is None:
            for param in self.params:
                parameter[param] = _unit_conv(getattr(self, param).value, getattr(self, param).unit, self._main_units[param], ref_period=None, ref_mintime=None, parameter_name=param, JD_convertable=getattr(self, param).JD_convertable) 
        else:
            parameter["dT"] = dT
            parameter["dP"] = dP
        return parameter["dP"] * x + parameter["dT"]
    
class Quad(model_component):
    """
    Quadratic O–C term: Q·epoch².

    Parameters:
      - Q (day/cycle² coefficient)
    """
    def __init__(self, params = {}, name="Quad", units={}):
        """
        Initializes a new instance of the Quad class with a specified quadratic coefficient and a name.

        Parameters:
        - params (dict): Dictionary containing parameter Q.
        - name (str): The name of the instance. Default is "Quad".
        """
        self._main_units = {"Q": "epoch"}
        self.params = {"Q":0} if params == {} else params
        self.name = name
        self.Ref_period = None
        self.set_params(params=params)


    def individual_model(self, x, Q=None):
        """
        Calculate the quadratic O–C term: Q·x².

        Parameters
        ----------
        x : float or array_like
            Cycle (epoch) value(s) at which to evaluate the model.
        Q : float, optional
            Quadratic coefficient. If None, uses this component’s stored `Q`.

        Returns
        -------
        float or ndarray
            Model prediction(s): `Q * x**2`.
        """
        # print("Q=",Q)
        parameter = {}
        if Q is None:
            for param in self.params:
                parameter[param] = _unit_conv(getattr(self, param).value, getattr(self, param).unit, self._main_units[param], ref_period=None, ref_mintime=None, parameter_name=param, JD_convertable=getattr(self, param).JD_convertable)  
        else:
            parameter["Q"] = Q
        return parameter["Q"] * x**2
    
class LiTE(model_component):
    """
    Light-Time Effect: analytical delay from orbital parameters.

    Parameters:
      - e       : eccentricity [0,1)
      - omega   : argument of periastron [deg]
      - P_LiTE  : period [epoch]
      - T_LiTE  : time of periastron [epoch, JD-convertable]
      - amp     : semi-amplitude [days]
    """
    def __init__(self, params={}, name="LiTE", units={}):
        self._main_units = {"e": "Unitless", "omega": "deg", "P_LiTE": "epoch", "T_LiTE": "epoch", "amp": "day"}
        self.params = {"e": 0, "omega": 0, "P_LiTE": 1, "T_LiTE": 0, "amp": 1} if params == {} else params
        if params != {}:
            sorted_items = sorted(self.params.items(), key=lambda item: self._main_units[item[0]])
            self.params = dict(sorted_items)
        self.name = name
        self.Ref_mintime = None
        self.Ref_period = None
        self.set_params(params)
        if hasattr(self, "T_LiTE"):
            self.T_LiTE.JD_convertable = True

    def individual_model(self, x, e=None, omega=None, P_LiTE=None, T_LiTE=None, amp=None, Ref_period=None, Ref_mintime=None):
        """
        Calculate the Light–Time Effect (LiTE) delay.

        Parameters
        ----------
        x : float or array_like
            Epoch(s) or time(s) at which to compute the delay.
        e : float, optional
            Orbital eccentricity (0 ≤ e < 1). If None, uses stored `e`.
        omega : float, optional
            Argument of periastron in degrees. If None, uses stored `omega`.
        P_LiTE : float, optional
            Orbital period in epochs. If None, uses stored `P_LiTE`.
        T_LiTE : float, optional
            Time of periastron passage in epochs. If None, uses stored `T_LiTE`.
        amp : float, optional
            Semi-amplitude of the LiTE in days. If None, uses stored `amp`.

        Returns
        -------
        ndarray
            Light-time delay(s) in days.
        """
        parameter = {}
        if e is None:
            self.Ref_period = Ref_period if Ref_period is not None else self.Ref_period
            self.Ref_mintime = Ref_mintime if Ref_mintime is not None else self.Ref_mintime
            for param in self.params:
                parameter[param] = _unit_conv(getattr(self, param).value, getattr(self, param).unit, self._main_units[param], ref_period=self.Ref_period, ref_mintime=self.Ref_mintime, parameter_name=param, JD_convertable=getattr(self, param).JD_convertable)
        else:
            parameter["e"] = e
            parameter["omega"] = omega
            parameter["P_LiTE"] = P_LiTE
            parameter["T_LiTE"] = T_LiTE
            parameter["amp"] = amp

        if isinstance(parameter["e"], float) and (parameter["e"] >= .999 or parameter["e"] < 0):
            # return an array of the same shape as x, filled with the sentinel value
            return np.full_like(x, -1e90)
        
        print("e=", parameter["e"])
        print("omega=", parameter["omega"])
        print("P_LiTE=", parameter["P_LiTE"])
        print("T_LiTE=", parameter["T_LiTE"])
        print("amp=", parameter["amp"])

        return Functions.calculate_lite_effect(
            x,
            parameter["e"],
            parameter["omega"],
            parameter["P_LiTE"],
            parameter["T_LiTE"],
            parameter["amp"],
        )
        
class LiTE_abspar(model_component):
    """
    LiTE variant using absolute parameters:

      - mass   [GM_sun]
      - P_LiTE [day]
      - ecc    [unitless]
      - omega  [deg]
      - T_LiTE [BJD]
      - inc    [deg]
    """
    def __init__(self, params = {}, name="LiTE_abspar"):
        self._main_units = {"mass": "GM_sun", "P_LiTE":"day", "ecc": "Unitless", "omega":"deg", "T_LiTE":"BJD", "inc":"deg"}
        self.params = {"mass":0, "P_LiTE":0, "ecc":0, "omega":0, "T_LiTE":0, "inc":90} if params == {} else params
        self.name = name
        self.Ref_period = None
        self.set_params(params=params)
        
class Grav_rad(model_component):
    """
    Radial gravitational term: c + b·epoch + a·epoch².

    Parameters:
      - a_grav, b_grav, c_grav [days]
    """
    def __init__(self, params = {}, name="Grav"):
        self._main_units = {"a_grav": "day", "b_grav":"day", "c_grav": "day"}
        self.params = {"a_grav":0, "b_grav":0, "c_grav":0} if params == {} else params
        self.name = name
        self.Ref_period = None
        self.set_params(params=params)

    def individual_model(self, x, a_grav=None, b_grav=None, c_grav=None, Ref_period=None, Ref_mintime=None):
        """
        Calculate the radial gravitational component: c + b·x + a·x².

        Parameters
        ----------
        x : float or array_like
            Cycle (epoch) value(s) at which to evaluate the model.
        a_grav : float, optional
            Quadratic coefficient (days). If None, uses stored `a_grav`.
        b_grav : float, optional
            Linear coefficient (days). If None, uses stored `b_grav`.
        c_grav : float, optional
            Constant offset (days). If None, uses stored `c_grav`.

        Returns
        -------
        float or ndarray
            Model prediction(s): `c_grav + b_grav*x + a_grav*x**2`.
        """
        parameter = {}
        # print("dT=", dT)
        # print("dP=", dP)
        if Ref_period is None:
            if hasattr(self, "Ref_period"):
                Ref_period = self.Ref_period  
        if a_grav is None:
            for param in self.params:
                parameter[param] = _unit_conv(getattr(self, param).value, getattr(self, param).unit, self._main_units[param], ref_period=None, ref_mintime=None, parameter_name=param, JD_convertable=getattr(self, param).JD_convertable) 
        else:
            parameter["a_grav"] = a_grav
            parameter["b_grav"] = b_grav
            parameter["c_grav"] = c_grav
        return parameter["c_grav"] + parameter["b_grav"] * x + parameter["a_grav"] * x**2
    
class Mag_act(model_component):
    """
    Magnetic activity term: c + A·sin(2π/Period·epoch + φ).

    Parameters:
      - P_mag [days], A_mag [days], phi [radians], c [days]
    """
    def __init__(self, params = {}, name="Mag"):
        self._main_units = {"P_mag": "epoch", "A_mag":"day", "phi": "epoch", "c": "epoch"}
        self.params = {"P_mag":0, "A_mag":0, "phi":0, "c":0} if params == {} else params
        self.name = name
        self.Ref_period = None
        self.set_params(params=params)

    def individual_model(self, x, P_mag=None, A_mag=None, phi=None, c=None, Ref_period=None, Ref_mintime=None):
        """
        Calculate the magnetic activity modulation: c + A_mag·sin(2π/ P_mag·x + φ).

        Parameters
        ----------
        x : float or array_like
            Cycle (epoch) value(s) at which to evaluate the model.
        P_mag : float, optional
            Modulation period in days. If None, uses stored `P_mag`.
        A_mag : float, optional
            Amplitude in days. If None, uses stored `A_mag`.
        phi : float, optional
            Phase offset in radians. If None, uses stored `phi`.
        c : float, optional
            Constant offset in days. If None, uses stored `c`.

        Returns
        -------
        float or ndarray
            Model prediction(s): `c + A_mag * sin(2π/ P_mag * x + phi)`.
        """
        parameter = {}
        # print("dT=", dT)
        # print("dP=", dP)
        if P_mag is None:
            for param in self.params:
                parameter[param] = _unit_conv(getattr(self, param).value, getattr(self, param).unit, self._main_units[param], ref_period=None, ref_mintime=None, parameter_name=param, JD_convertable=getattr(self, param).JD_convertable) 
        else:
            parameter["P_mag"] = P_mag
            parameter["A_mag"] = A_mag
            parameter["phi"] = phi
            parameter["c"] = c
        return parameter["c"] + parameter["A_mag"] * np.sin(2 * np.pi / parameter["P_mag"] * x  + (parameter["phi"]))
        
def calculate_aic_bic(model, data, print_results=False):
    """
    Compute goodness‐of‐fit metrics for a fitted model.

    Parameters
    ----------
    model : object
        A fit‐object whose `.model_components`  list and `.calculate_oc()`
        method produce the model O–C values.
    data : object
        A data container with attributes:
          - `.Ecorr` : array of epochs
          - `.OC`    : array of observed-minus-calculated values
          - `.Errors`: array of observational uncertainties
    print_results : bool, optional
        If True, print Chi², reduced Chi², AIC, and BIC to stdout.

    Returns
    -------
    chi2 : float
        Sum of squared residuals divided by variances.
    reduced_chi2 : float
        chi2 / (N − k), where N is number of data points and k is number of free parameters.
    AIC : float
        Akaike Information Criterion = χ² + 2·k.
    BIC : float
        Bayesian Information Criterion = χ² + ln(N)·k.
    """
    model = copy.deepcopy(model)
    data  = copy.deepcopy(data)
    model.epochs = data.Ecorr

    fit2 = fit(data=data, model=model)
    fit2.fitted_model = fit2.model

    # count free parameters
    param_count=0
    for mc in model.model_components:
        for p_name, value in mc.params.items():
            if value.vary:
                param_count += 1

    for mc in fit2.model.model_components:
        for p_name, value in mc.params.items():
            value.vary = False

    chi2 = np.sum((fit2.total_oc_delay([], fit2.data.m1, fit2.data.m2, fit2.data.inc, mintimes_in_data=True) - data.OC) ** 2 / data.Errors ** 2)
    reduced_chi2 = chi2 / (len(data.OC) - param_count)
    AIC = chi2 + 2 * param_count
    BIC = chi2 + np.log(len(data.OC)) * param_count

    if print_results:
        print(f"Model: {model.name}")
        print(f"Chi-square: {chi2:.2f}")
        print(f"Reduced Chi-square: {reduced_chi2:.2f}")
        print(f"AIC: {AIC:.2f}")
        print(f"BIC: {BIC:.2f}")

    return chi2, reduced_chi2, AIC, BIC
    
# funcions
class Functions:
    @staticmethod
    def calculate_lite_effect(times, e, omega_deg, P_LiTE, T_LiTE, amp):
        """
        Calculates the Light–Time Effect (LiTE) delay given full orbital elements.

        Parameters
        ----------
        times : array_like
            Observation times (same units as P_LiTE & T_LiTE, e.g. days).
        e : float
            Orbital eccentricity (0 <= e < 1).
        omega_deg : float
            Argument of periastron, in degrees.
        P_LiTE : float
            Orbital period (same units as `times`).
        T_LiTE : float
            Time of periastron passage (same units as `times`).
        amp : float
            LiTE semi-amplitude (same units as output delay).

        Returns
        -------
        delay : np.ndarray
            Light-time delays, same shape as `times`.
        """
        times = np.asarray(times, dtype=float)

        # sanity-check eccentricity
        if not (0.0 <= e < 1.0):
            return -np.inf
        if not (0 <= amp):
            return -np.inf
        if not (0 <= P_LiTE):
            return -np.inf

        # mean anomaly M in [0,2π)
        M = np.remainder(2*np.pi * (times - T_LiTE) / P_LiTE, 2*np.pi)

        # solve Kepler's equation for eccentric anomaly E
        E = Functions._kepler(M, e)

        # true anomaly v
        v = 2 * np.arctan2(
            np.sqrt(1 + e) * np.sin(E / 2),
            np.sqrt(1 - e) * np.cos(E / 2)
        )

        # convert argument of periastron to radians
        ω = np.radians(omega_deg)

        # guard against small denominator in the projection factor
        denom = np.sqrt(1 - e**2 * np.cos(ω)**2)
        if np.isclose(denom, 0.0):
            raise ValueError("Denominator in scale factor too small (check e and ω).")

        # scale amplitude by projection factor
        scale = amp / denom

        # LiTE delay formula
        delay = scale * (
            (1 - e**2) / (1 + e * np.cos(v)) * np.sin(v + ω)
            + e * np.sin(ω)
        )

        return delay

    @staticmethod
    def _kepler(M, e, tolerance=1e-12, max_iter=10000):
        """
        Solve Kepler’s equation M = E - e·sin(E) for the eccentric anomaly E
        via element-wise Newton–Raphson, stopping each entry once converged.

        Parameters
        ----------
        M : array_like
            Mean anomaly in radians.
        e : float
            Eccentricity (0 <= e < 1).
        tolerance : float
            Convergence threshold for |ΔE| per entry.
        max_iter : int
            Maximum number of Newton iterations.

        Returns
        -------
        E : np.ndarray
            Eccentric anomaly in radians, same shape as M.
        """
        if not (0.0 <= e < 1.0):
            raise ValueError(f"Eccentricity e={e} out of bounds [0,1).")

        M = np.asarray(M, dtype=np.float64)
        E = M.copy()                       # initial guess = mean anomaly
        mask = np.ones_like(E, dtype=bool) # True = entries still unconverged

        for _ in range(max_iter):
            # compute delta only for unconverged entries
            delta = np.zeros_like(E)
            denom = 1.0 - e * np.cos(E[mask])
            delta[mask] = (
                (E[mask] - e * np.sin(E[mask]) - M[mask])
                / denom
            )

            # update those entries
            E[mask] -= delta[mask]

            # mark entries that have now converged
            mask[mask] = np.abs(delta[mask]) > tolerance

            # if all entries are converged, stop early
            if not mask.any():
                break
        else:
            # optional: warn if full iterations completed with some entries still unconverged
            unconverged = np.sum(mask)
            if unconverged:
                import warnings
                warnings.warn(
                    f"{unconverged} points in _kepler did not converge after {max_iter} iterations."
                )

        return E

    
def _unit_conv(value, from_unit, to_unit, ref_period=None, ref_mintime=None, parameter_name=None, JD_convertable=False):
    """
    Convert a scalar or array between units, handling “epoch” and Julian‐date offsets.

    Parameters
    ----------
    value : float or ndarray
        The quantity to convert.
    from_unit : str
        Original unit (e.g. "day", "epoch", "BJD", etc.).
    to_unit : str
        Desired unit.
    ref_period : float, optional
        Reference period in same units as time for epoch↔time conversions.
    ref_mintime : float, optional
        Reference minimum time (e.g. JD offset) for epoch↔time conversions.
    parameter_name : str, optional
        Name of the parameter (used in error messages).
    JD_convertable : bool, optional
        Whether “epoch”↔Julian‐date conversion is allowed.

    Returns
    -------
    float or ndarray
        Converted value(s).
    """
    if from_unit == to_unit:
        return value
    elif to_unit != "epoch" and from_unit != "epoch":
        return _unit_conv_nonepoch(value, from_unit, to_unit)
    elif (to_unit == "epoch") and ref_period is None:
        raise ValueError(f"Ref_period is required for converting the unit named '{parameter_name}' to epoch.")
    elif (from_unit == "epoch") and ref_period is None:
        raise ValueError(f"Ref_period is required for converting the unit named '{parameter_name}' from epoch.")
    elif from_unit=="BJD" or to_unit=="BJD" or from_unit=="HJD" or to_unit=="HJD":
        if to_unit == "epoch":
            if ref_mintime is not None:
                con_unit = (value - ref_mintime) / ref_period
                return con_unit
            else:
                raise ValueError(f"Ref_mintime is required for converting the unit named '{parameter_name}' to epoch.")
        elif from_unit == "epoch":
            if ref_mintime is not None:
                con_unit = value*ref_period + ref_mintime
                return con_unit
            else:
                raise ValueError(f"Ref_mintime is required for converting the unit named '{parameter_name}' to epoch.")
    elif to_unit == "epoch":
        unit = u.Unit(from_unit)
        value_with_unit = value * unit
        day_unit = value_with_unit.to(u.Unit("day"))
        con_unit = day_unit / ref_period
    elif from_unit == "epoch":
        day_value = value * ref_period
        value_with_unit = day_value * u.Unit("day")
        con_unit = value_with_unit.to(u.Unit(to_unit))
    return con_unit.value

def _unit_conv_nonepoch(value, from_unit, to_unit):
    unit = u.Unit(from_unit)
    value_with_unit = value * unit
    con_unit = value_with_unit.to(u.Unit(to_unit))
    return con_unit.value
    
def fit_plot(model=None,
             data=None,
             samples=None,
             nrandom_samples=0,
             outfile_tag="",
             save_plots=False,
             show=False,
             other_models=None,
             other_model_colors=None,   # <-- NEW parameter
             model_color="r",           # <-- NEW parameter
             color_palette="tab20",
             group_colors=None,
             group_shapes=None,
             group_sizes=None,
             x_axis_bot="epoch",
             x_axis_top=None,
             y_axis_left="second",
             y_axis_right=None,
             extend_graph_factor=0.05,
             extend_graph_factor_negx=None,
             extend_graph_factor_posx=None,
             graph_size=None,
             label_size=16,
             # Tick size parameters:
             tick_size=10,
             tick_size_x_bottom=None,
             tick_size_x_top=None,
             tick_size_y_left=None,
             tick_size_y_right=None,
             legend_size=10,
             legend_shape=(5,6),
             x_lim=None,
             y_lim=None,
             y_lim_right=None,
             legend_position="best",
             draw_legend=True,
             bjd_offset=2450000.0,
             res_plot=False,
             res_height_ratios=(4, 1),
             draw_main_model=True,
             res_hspace=0.2,
             res_ylim=None,
             integrator="IAS15"):
    """
    Plots an O–C (Observed minus Calculated) diagram with error bars, a median fit,
    and optionally a residuals subplot. Top‐axis year ticks follow your January 1 rule.

    Parameters
    ----------
    model : fit
        The fit object containing `.model_components` and `.calculate_oc()`.
    data : object
        Data container with attributes:
          - .df             : pandas.DataFrame with columns "Ecorr", "OC", "Errors", "Data_group"
          - .Ecorr, .Mintimes : array-like epochs and corresponding times
          - .Ref_period, .Ref_mintime : floats for epoch⇄time conversions
          - .object_name    : str used for output filenames
    samples : ndarray, optional
        MCMC sample array of shape (n_samples, n_params); used for uncertainty envelopes.
    nrandom_samples : int, default=0
        Number of random posterior curves to overplot.
    outfile_tag : str, default=""
        Suffix for saved plot filenames.
    save_plots : bool, default=False
        If True, saves the figure as a PNG.
    show : bool, default=False
        If True, calls `plt.show()`, otherwise closes the figure.
    other_models : list of model_component, optional
        Additional models to overlay (each must have `.model_components`).
    other_model_colors : list of color-spec, optional
        Colors to use when overplotting each model in `other_models`.
        If provided, length must match `len(other_models)`.
    model_color : color-spec, default="r"
        Color for the main median-fit curve.
    color_palette : str or Colormap, default="tab20"
        Matplotlib palette for grouping.
    group_colors : list or str, optional
        Explicit colors for each Data_group.
    group_shapes : list or str, optional
        Marker shapes for each Data_group.
    group_sizes : list or int, optional
        Marker sizes for each Data_group.
    x_axis_bot : {"epoch","bjd","year"} or str, default="epoch"
        Bottom x-axis label type.
    x_axis_top : {"epoch","bjd","year"} or str, optional
        Top x-axis label type.
    y_axis_left : {"day","hour","minute","second","millisecond"} or str, default="second"
        Unit for left y-axis.
    y_axis_right : {"day","hour","minute","second","millisecond"} or str, optional
        Unit for right y-axis.
    extend_graph_factor : float, default=0.05
        Fractional extension of x-range beyond data for model plotting.
    graph_size : tuple, optional
        Figure size (width, height) in inches; defaults to (14,8) or (14,14) if `res_plot`.
    label_size : int, default=16
        Font size for axis labels.
    tick_size : int, default=10
        Base tick label size.
    tick_size_x_bottom, tick_size_x_top, tick_size_y_left, tick_size_y_right : int, optional
        Specific tick label sizes; default to `tick_size` if None.
    legend_size : int, default=10
        Font size for legend text.
    legend_shape : tuple, default=(5,6)
        Legend grid layout (columns, rows).
    x_lim : tuple, optional
        Manual x-axis limits.
    y_lim, y_lim_right : tuple, optional
        Manual y-axis limits for left and right axes.
    legend_position : {"best","top","bottom","left","right"} or tuple, default="best"
        Legend location.
    draw_legend : bool, default=True
        Whether to draw the legend.
    bjd_offset : float, default=2450000.0
        Offset subtracted when labeling BJD axis.
    res_plot : bool, default=False
        If True, include a residuals subplot below the main plot.
    res_height_ratios : tuple, default=(4,1)
        Height ratio of main plot to residuals subplot.
    draw_main_model : bool, default=True
        If True, overplot the median-fit curve.
    res_hspace : float, default=0.2
        Vertical spacing between main and residuals axes.
    res_ylim : tuple or None, default=None
        If provided, sets the y-limits of the residuals subplot to this (min, max).
    integrator : str, default="IAS15"
        Name of the integrator passed to `fit(model, data, integrator=...)`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import copy
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator, FixedLocator, FixedFormatter

    # Set tick sizes
    tick_size_x_bottom = tick_size_x_bottom or tick_size
    tick_size_x_top    = tick_size_x_top or tick_size
    tick_size_y_left   = tick_size_y_left or tick_size
    tick_size_y_right  = tick_size_y_right or tick_size

    # Determine figure size
    if graph_size is None:
        graph_size = (14, 14) if res_plot else (14, 8)

    # Formatting helpers
    def dynamic_format(val_range):
        if val_range <= 1e-5:  return "%.7f"
        if val_range <= 1e-4:  return "%.6f"
        if val_range <= 1e-3:  return "%.5f"
        if val_range <= 1e-2:  return "%.4f"
        if val_range <= 0.1:   return "%.3f"
        if val_range <= 1:     return "%.2f"
        if val_range <= 100:   return "%.1f"
        return "%.0f"

    def approximate_year_from_bjd(bjd):
        return 2000.0 + (bjd - 2451545.0) / 365.25

    def approximate_bjd_from_year(year):
        return 2451545.0 + (year - 2000.0) * 365.25

    def _apply_bottom_xaxis(ax, xtype):
        ax.tick_params(axis='x', labelsize=tick_size_x_bottom)
        if xtype.lower() == "bjd":
            ref_mintime = data.Ref_mintime
            ref_period  = data.Ref_period
            ticks = ax.get_xticks()
            bjds = [ref_mintime + t*ref_period for t in ticks]
            diffs = [b - bjd_offset for b in bjds]
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FixedFormatter([f"{d:.0f}" for d in diffs]))
            ax.set_xlabel(f"BJD - {int(bjd_offset)}", fontsize=label_size)
        elif xtype.lower() == "epoch":
            ax.set_xlabel("Cycle", fontsize=label_size)
        elif xtype.lower() == "year":
            ax.set_xlabel("Year", fontsize=label_size)
        else:
            ax.set_xlabel(xtype, fontsize=label_size)

    # Unit conversion factors
    conv = {"day":1, "hour":24, "minute":1440, "second":86400, "millisecond":86400000}
    factor_left  = conv[y_axis_left.lower()]
    factor_right = conv[y_axis_right.lower()] if y_axis_right else None

    # Prepare x-range for model curves
    e_min, e_max = min(data.Ecorr), max(data.Ecorr)
    if extend_graph_factor_negx is None:
        extend_graph_factor_negx = extend_graph_factor
    if extend_graph_factor_posx is None:
        extend_graph_factor_posx = extend_graph_factor

    egap = (e_max - e_min)
    ep = np.linspace(e_min - egap * extend_graph_factor_negx,
                     e_max + egap * extend_graph_factor_posx, 3000)
    t_min, t_max = min(data.Mintimes), max(data.Mintimes)
    tgap = (t_max - t_min)
    new_Mintimes = np.linspace(t_min - tgap * extend_graph_factor_negx,
                               t_max + tgap * extend_graph_factor_posx, 3000)

    # Plot setup
    if res_plot:
        fig, (ax_main, ax_res) = plt.subplots(
            2, 1, sharex=True,
            gridspec_kw={'height_ratios':res_height_ratios, 'hspace':res_hspace},
            figsize=graph_size
        )
        ax_left = ax_main
    else:
        fig, ax_left = plt.subplots(figsize=graph_size)

    # Data groups
    df0 = data.df.reset_index(drop=True)
    groups = list(df0.groupby("Data_group"))
    cmap = plt.get_cmap(color_palette, len(groups))
    for i, (name, grp) in enumerate(groups):
        color = (group_colors[i] if group_colors else cmap(i))
        marker = (group_shapes[i] if group_shapes else ".")
        size   = (group_sizes[i] if group_sizes else 5)
        mfc = 'none' if marker == "o" else color
        ax_left.errorbar(
            grp["Ecorr"], grp["OC"]*factor_left, yerr=grp["Errors"]*factor_left,
            fmt=marker, markersize=size, mfc=mfc, elinewidth=1.2, capsize=2,
            color=color, label=name
        )

    # Perform fit
    fit2 = fit(model, data, integrator=integrator)
    fit2 = copy.deepcopy(fit2)
    fit2.model.Ref_period  = fit2.data.Ref_period
    fit2.model.Ref_mintime = fit2.data.Ref_mintime
    fit2.fitted_model = fit2.model.fix_units()
    fit2.model        = fit2.model.fix_units()

    # Draw uncertainty bands or median-only
    if samples is not None:
        MAX_DRAW = 500
        if samples.shape[0] > MAX_DRAW:
            idx = np.random.default_rng(0).choice(samples.shape[0], MAX_DRAW, replace=False)
            samp = samples[idx]
        else:
            samp = samples

        sigma_p, sigma_m = fit2.calculate_uncertanities(samp, epochs=ep, mintimes=new_Mintimes)
        for comp in fit2.model.model_components:
            for p in comp.params.values():
                p.vary = False
        best = fit2.total_oc_delay([], fit2.data.m1, fit2.data.m2, fit2.data.inc, ep)
        best_left = best * factor_left

        ax_left.fill_between(ep,
                             best_left + sigma_p*factor_left,
                             best_left - sigma_m*factor_left,
                             color="gray", alpha=0.15, label="±1 σ")

        if nrandom_samples > 0:
            theta0 = [p.value for comp in model.model_components for p in comp.params.values()]
            free_idx = [i for i,(p) in enumerate(
                [p for comp in model.model_components for p in comp.params.values()]) if p.vary]
            samp_idx = np.random.default_rng().choice(samp.shape[0],
                                                     min(nrandom_samples, len(samp)),
                                                     replace=False)
            for row in samp[samp_idx]:
                th = np.array(theta0)
                th[free_idx] = row
                curve = (fit2.total_oc_delay(th, fit2.data.m1, fit2.data.m2,
                                             fit2.data.inc, ep, fix_units_first=True)
                         * factor_left)
                ax_left.plot(ep, curve, lw=0.6, alpha=0.15, color="gray", zorder=-1)
            ax_left.plot([], [], color="gray", alpha=0.15, lw=0.6,
                         label=f"{len(samp_idx)} random samples")
    else:
        for comp in fit2.model.model_components:
            for p in comp.params.values():
                p.vary = False
        best = fit2.total_oc_delay([], fit2.data.m1, fit2.data.m2, fit2.data.inc, ep)
        best_left = best * factor_left

    # Median-fit line
    if draw_main_model:
        ax_left.plot(
            ep, best_left,
            color=model_color,      # use the new model_color
            linewidth=3,
            label="Median fit",
            zorder=100
        )
    ax_left.axhline(0, color="black", linestyle="--", alpha=0.4)

    # Main plot limits
    ax_left.set_xlim(x_lim if x_lim else (e_min - egap*extend_graph_factor_negx,
                                         e_max + egap*extend_graph_factor_posx))
    if y_lim:
        ax_left.set_ylim(y_lim)
    else:
        dmin, dmax = (data.df["OC"]*factor_left).agg(["min","max"])
        if dmin == dmax:
            dmin -= 1e-7; dmax += 1e-7
        pad = (dmax - dmin) * 0.05
        ax_left.set_ylim(dmin - pad, dmax + pad)

    # Right y-axis
    if y_axis_right:
        ax_r = ax_left.twinx()
        if y_lim_right:
            ax_r.set_ylim(y_lim_right)
        else:
            lmin, lmax = ax_left.get_ylim()
            ratio = factor_right / factor_left
            ax_r.set_ylim(lmin * ratio, lmax * ratio)
        ax_r.yaxis.set_major_locator(MaxNLocator(nbins=7))
        rf = dynamic_format(ax_r.get_ylim()[1] - ax_r.get_ylim()[0])
        ax_r.yaxis.set_major_formatter(FormatStrFormatter(rf))
        ax_r.tick_params(axis='y', labelsize=tick_size_y_right)
        ax_r.set_ylabel(f"O-C ({y_axis_right.title()})", fontsize=label_size)

    # Left y formatting
    ax_left.yaxis.set_major_locator(MaxNLocator(nbins=7))
    lf = dynamic_format(ax_left.get_ylim()[1] - ax_left.get_ylim()[0])
    ax_left.yaxis.set_major_formatter(FormatStrFormatter(lf))
    ax_left.tick_params(axis='y', labelsize=tick_size_y_left)
    ax_left.set_ylabel(f"O-C ({y_axis_left.title()})", fontsize=label_size)

    # Bottom x-axis
    _apply_bottom_xaxis(ax_left, x_axis_bot)

    # Top x-axis
    if x_axis_top:
        ax_top = ax_left.twiny()
        ax_top.set_xlim(ax_left.get_xlim())
        ax_top.tick_params(axis='x', labelsize=tick_size_x_top)
        ax_top.xaxis.set_ticks_position('top')
        if x_axis_top.lower() == "year":
            x0, x1 = ax_left.get_xlim()
            b0 = data.Ref_mintime + x0*data.Ref_period
            b1 = data.Ref_mintime + x1*data.Ref_period
            y0 = approximate_year_from_bjd(b0)
            y1 = approximate_year_from_bjd(b1)
            t0 = int(np.floor(y0)) + 1
            t1 = int(np.ceil(y1))
            for d in (8, 7, 6):
                if (t1 - t0) % d == 0:
                    div = d; break
            else:
                div = None
            while div is None and t1 > t0:
                t1 -= 1
                for d in (8, 7, 6):
                    if (t1 - t0) % d == 0:
                        div = d; break
            if div:
                step = (t1 - t0) // div
                yrs = [t0 + i*step for i in range(div+1)]
            else:
                yrs = [t0, t1]
            pos = []
            for yr in yrs:
                bj = approximate_bjd_from_year(yr)
                pos.append((bj - data.Ref_mintime) / data.Ref_period)
            ax_top.xaxis.set_major_locator(FixedLocator(pos))
            ax_top.xaxis.set_major_formatter(FixedFormatter([str(yr) for yr in yrs]))
            ax_top.set_xlabel("Year", fontsize=label_size)
        elif x_axis_top.lower() == "bjd":
            ax_top.set_xlabel("BJD", fontsize=label_size)
        else:
            ax_top.set_xlabel(x_axis_top.title(), fontsize=label_size)

    # Overlay other_models
    if other_models:
        fitb = copy.deepcopy(fit2)
        for idx, om in enumerate(other_models):
            om_copy = copy.deepcopy(om)
            fitb.model = om_copy
            for mc in om_copy.model_components:
                for p in mc.params.values():
                    p.vary = False
            bf = (fitb.total_oc_delay([], fitb.data.m1, fitb.data.m2,
                                      fitb.data.inc, ep, fix_units_first=True)
                  * factor_left)

            # pick color for this model if provided
            color = None
            if other_model_colors:
                try:
                    color = other_model_colors[idx]
                except IndexError:
                    color = None

            ax_left.plot(
                ep, bf,
                label=om.name,
                linewidth=1,
                zorder=50,
                color=color
            )

    # Legend
    if draw_legend:
        ax_left.legend(loc=legend_position if isinstance(legend_position, str) else "best",
                       bbox_to_anchor=legend_position if isinstance(legend_position, tuple) else None,
                       fontsize=legend_size, ncol=legend_shape[0],
                       frameon=False, title="Data Group")

    # Residuals subplot
    if res_plot:
        obs = df0["OC"].values * factor_left
        pred = np.interp(df0["Ecorr"].values, ep, best_left)
        res = obs - pred
        err = df0["Errors"].values * factor_left

        for name, grp in df0.groupby("Data_group"):
            idx_grp = grp.index
            ax_res.errorbar(
                grp["Ecorr"], res[idx_grp], yerr=err[idx_grp],
                fmt=group_shapes[0] if group_shapes else ".",
                markersize=group_sizes[0] if group_sizes else 5,
                elinewidth=1.2, capsize=2,
                color=group_colors[0] if group_colors else cmap(0)
            )
        ax_res.axhline(0, linestyle='--', linewidth=1)
        _apply_bottom_xaxis(ax_res, x_axis_bot)

        if res_ylim:
            ax_res.set_ylim(res_ylim)
        else:
            rmin, rmax = res.min(), res.max()
            if rmin == rmax:
                rmin -= 1e-7; rmax += 1e-7
            pad = (rmax - rmin)*0.05
            ax_res.set_ylim(rmin - pad, rmax + pad)

        ax_res.yaxis.set_major_locator(MaxNLocator(nbins=7))
        rf = dynamic_format(ax_res.get_ylim()[1] - ax_res.get_ylim()[0])
        ax_res.yaxis.set_major_formatter(FormatStrFormatter(rf))
        ax_res.tick_params(axis='y', labelsize=tick_size_y_left)
        ax_res.set_ylabel(y_axis_left.title(), fontsize=label_size)

        if y_axis_right:
            ax_res_r = ax_res.twinx()
            l0, l1 = ax_res.get_ylim()
            ax_res_r.set_ylim(l0*factor_right/factor_left, l1*factor_right/factor_left)
            ax_res_r.yaxis.set_major_locator(MaxNLocator(nbins=7))
            rf2 = dynamic_format(ax_res_r.get_ylim()[1] - ax_res_r.get_ylim()[0])
            ax_res_r.yaxis.set_major_formatter(FormatStrFormatter(rf2))
            ax_res_r.tick_params(axis='y', labelsize=tick_size_y_right)
            ax_res_r.set_ylabel(y_axis_right.title(), fontsize=label_size)

        if x_lim:
            ax_res.set_xlim(x_lim)
        else:
            ax_res.set_xlim(e_min - egap*extend_graph_factor_negx,
                            e_max + egap*extend_graph_factor_posx)
        ax_res.set_title("Residuals", fontsize=label_size)

    # Save or show
    if save_plots:
        plt.savefig(f"{data.object_name}_errorbar_plot_{outfile_tag}.png",
                    dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
