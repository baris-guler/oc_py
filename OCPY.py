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
        mass3(mf, mf_std, m1, m1_std, m2, m2_std, inc=90, inc_std=0.01): 
            Computes the minimum companion mass based on the binary mass function.
    """

    def __init__(self, epochs=np.array([]), name="OC_Model", model_components: list = None, 
                 Ref_mintime=None, Ref_period=None, nan_policy='raise'):
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
    def __init__(self, object_name="undefined", Mintimes=np.array([]), Mintypes=np.array([]), Errors=np.array([]), Units=np.array([]), Ref_mintime=None, Ref_period=None, Data_group=np.array([]), Weights=np.array([]), data_file=None):
        self.object_name = object_name
        self._calculated = False
        self.binned = False
        self.m1 = None
        self.m2 = None
        self.inc = None
        if data_file is not None:
            self._read_data_file(data_file)
        datas = ["Mintimes", "Mintypes", "Errors", "Units", "Ref_mintime", "Ref_period", "Data_group"]
        for data in datas:
            if not hasattr(self, data):
                setattr(self, data, locals().get(data))
        
        if self.Errors.size < 1:
            self.Errors = np.full(self.Mintimes.shape, 1)
        
        if self.Mintimes.size > 0 and self.Ref_mintime is not None and self.Ref_period is not None:
            self.Ecorr, self.OC = self.generate_oc()
        else:
            self.Ecorr = np.array([])
            self.OC = np.array([])
            
        if self.Data_group.size > 0:
            self.Data_group = np.where(pd.isnull(self.Data_group), 'None', self.Data_group)
        else:
            self.Data_group = np.full(self.Mintimes.shape, "None")
            
        if self.Units.size == 0:
            self.Units = np.full(self.Mintimes.shape, "Default")
            
        self.Weights = Weights
        if self.Weights.size == 0:
            self.fill_weights()
            
        self.df = pd.DataFrame({
            'Mintimes': self.Mintimes,
            'Mintypes': self.Mintypes,
            'Errors': self.Errors,
            'Units': self.Units,
            'Data_group': self.Data_group,
            'Ecorr': self.Ecorr,
            'OC': self.OC,
            'Weights': self.Weights
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
                    
    def remove_data(self, values_x_min=None, values_x_max=None, values_y_min=None, values_y_max=None, data_groups=None):
        new_data = copy.deepcopy(self)
        if values_x_min is not None:
            new_data.df = new_data.df[new_data.df["Ecorr"] > values_x_min]
        if values_x_max is not None:
            new_data.df = new_data.df[new_data.df["Ecorr"] < values_x_max]
        if values_y_min is not None:
            new_data.df = new_data.df[new_data.df["OC"] > values_y_min]
        if values_y_max is not None:
            new_data.df = new_data.df[new_data.df["OC"] < values_y_max]
        if data_groups is not None and isinstance(data_groups, str):
            new_data.df = new_data.df[new_data.df["Data_group"] != data_groups]
        elif data_groups is not None and isinstance(data_groups, list):
            new_data.df = new_data.df[~new_data.df["Data_group"].isin(data_groups)]
        return new_data
            
    def fill_weights(self, value=None):
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
        Plots outliers
        """
        Ecorr_outlier = self.Ecorr[outliers]
        OC_Outlier = self.OC[outliers]
        Ecorr_res = self.Ecorr[~outliers]
        OC_res = self.OC[~outliers]
        for v_line in v_lines:
            plt.axvline(v_line, color="r", linestyle="--", label="Vertical Limits")
        for h_line in h_lines:
            plt.axhline(h_line, color="r", linestyle="--", label="Horizontal Limits")
        plt.xlabel("Ecorr")
        plt.ylabel("OC")
        plt.legend
        if len(Ecorr_outlier) > 0:
            plt.plot(Ecorr_outlier, OC_Outlier, "rx", label="Outliers")
        plt.plot(Ecorr_res, OC_res, "b.", label="res")
        plt.legend()
        plt.show()
        
    def sigma_outliers(self, treshold=3, plot=False, OC=None, additional_method = None, additional_params = None):
        """
        Identifies outliers using sigma-clipping method.

        Args:
            treshold (float): Number of standard deviations to consider as threshold.
            plot (bool): Whether to plot outliers.
            OC (numpy.ndarray): Array of observed minus calculated (O-C) values.
            additional_method (str): Additional method for outlier detection. Can be 'moving_window, 'fixed_window' or 'spline'.
            additional_params (dict): Additional parameters for additional_method.

        Returns:
            numpy.ndarray: Boolean array indicating outliers.
        """
        if additional_method is not None:
            outliers = self._addm(additional_method, additional_params, main="sigma_outliers", main_params=[treshold], plot=plot)
            return outliers
        OC = self.OC if OC is None else OC
        mean_oc = np.mean(OC)
        std_oc = np.std(OC)
        condition = (OC < mean_oc + treshold*std_oc) & (OC > mean_oc - treshold*std_oc)
        if plot:
            self._plot_outliers(~condition, h_lines=[mean_oc + treshold*std_oc, mean_oc - treshold*std_oc])
        return ~condition
    
    def zscore_outliers(self, treshold=3, plot=False, OC=None, additional_method = None, additional_params = None):
        """
        Identifies outliers using z-score method.

        Args:
            treshold (float): Number of standard deviations to consider as threshold.
            plot (bool): Whether to plot outliers.
            OC (numpy.ndarray): Array of observed minus calculated (O-C) values.
            additional_method (str): Additional method for outlier detection. Can be 'moving_window, 'fixed_window' or 'spline'.
            additional_params (dict): Additional parameters for additional_method.

        Returns:
            numpy.ndarray: Boolean array indicating outliers.
        """
        from scipy.stats import zscore
        if additional_method is not None:
            outliers = self._addm(additional_method, additional_params, main="zscore_outliers", main_params=[treshold], plot=plot)
            return outliers
        OC = self.OC if OC is None else OC
        z_scores = zscore(OC)
        condition = np.abs(z_scores) < treshold
        if plot:
            self._plot_outliers(~condition)
        return ~condition
    
    def box_outliers(self, plot=False, treshold=1.5, OC=None, additional_method = None, additional_params = None):
        """
        Identifies outliers using boxplot method.

        Args:
            plot (bool): Whether to plot outliers.
            treshold (float): Multiplier for the interquartile range to consider as threshold.
            OC (numpy.ndarray): Array of observed minus calculated (O-C) values.
            additional_method (str): Additional method for outlier detection. Can be 'moving_window, 'fixed_window' or 'spline'.
            additional_params (dict): Additional parameters for additional_method.

        Returns:
            numpy.ndarray: Boolean array indicating outliers.
        """
        if additional_method is not None:
            outliers = self._addm(additional_method, additional_params, main="zscore_outliers", main_params=[treshold], plot=plot)
            return outliers
        OC = self.OC if OC is None else OC
        q1 = np.percentile(OC, 25)
        q3 = np.percentile(OC, 75)
        iqr = q3 - q1
        lower_threshold = q1 - treshold * iqr
        upper_threshold = q3 + treshold * iqr
        outliers = np.logical_or(OC < lower_threshold, OC > upper_threshold)
        if plot:
            self._plot_outliers(outliers, h_lines=[lower_threshold, upper_threshold])
        return outliers
    
    def chauvenet_outliers(self, plot=False, treshold=0.5, OC=None, additional_method = None, additional_params = None):
        """
        Identifies outliers using Chauvenet's criterion method.

        Args:
            plot (bool): Whether to plot outliers.
            treshold (float): Threshold probability for Chauvenet's criterion.
            OC (numpy.ndarray): Array of observed minus calculated (O-C) values.
            additional_method (str): Additional method for outlier detection.
            additional_params (dict): Additional parameters for additional_method.

        Returns:
            numpy.ndarray: Boolean array indicating outliers.
        """
        if additional_method is not None:
            outliers = self._addm(additional_method, additional_params, main="zscore_outliers", main_params=[treshold], plot=plot)
            return outliers
        from scipy import stats as st
        OC = self.OC if OC is None else OC
        mean = OC.mean()
        std = OC.std()
        d = ((OC - mean) / std)
        possibility = st.norm.cdf(d)
        criteria = (1 - possibility) * len(OC)
        outliers = criteria < treshold
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
                outliers = self._moving_window(window_rate=additional_params["window_rate"], window_step_rate=additional_params["window_step_rate"], window_size=additional_params["window_size"], method=main, window_start=additional_params["window_start"], treshold=main_params[0], plot=plot)
            elif additional_method == "fixed_window":
                outliers = self._fixed_window(window_rate=additional_params["window_rate"], window_size=additional_params["window_size"], window_count=additional_params["window_count"], window_start=additional_params["window_start"], method=main, treshold=main_params[0], plot=plot)
            elif additional_method == "spline":
                outliers = self._spline(smoothing=additional_params["smoothing"], degree=additional_params["degree"], method=main, treshold=main_params[0], plot=plot)
        return outliers

    def _moving_window(self, window_rate=.1, window_step_rate=.01, window_size=None, method="sigma_outliers", window_start=None, treshold=3, plot=False):
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
            outliers = np.concatenate((outliers, OC_window[method(treshold=treshold, OC=OC_window)]))
            window_start += dif * window_step_rate
            window_end += dif * window_step_rate
        outliers = np.unique(outliers)
        outliers = np.isin(np.arange(len(self.OC)), np.where(np.isin(self.OC, outliers)))
        if plot:
            self._plot_outliers(outliers)
        return outliers
    
    def _fixed_window(self, window_rate=.1, window_size=None, window_count=None, window_start=None,  method="sigma_outliers", treshold=3, plot=False):
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
            outliers = np.concatenate((outliers, array[method(treshold=treshold, OC=array)]))
        outliers = np.isin(np.arange(len(self.OC)), np.where(np.isin(self.OC, outliers)))
        if plot:
            for i in split_points:
                plt.axvline(i, alpha=.3, linestyle="--")
            self._plot_outliers(outliers)
        return outliers

    def _spline(self, smoothing=1, degree=3, method="sigma_outliers", treshold=3, plot=False):
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

        outliers = method(treshold=treshold, OC=OC_dif)

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
            median_error = .1
            return median_Ecorr, median_oc, median_error

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
        from scipy.signal import lombscargle
        frequencies = np.linspace(min_freq, max_freq, num_frequencies)
        angular_frequencies = 2 * np.pi * frequencies  

        pgram = lombscargle(self.Ecorr, self.OC, angular_frequencies)

        pgram_normalized = pgram / (len(self.Ecorr) / 2)
        if plot:
            plt.plot(frequencies, pgram_normalized)
            plt.xlabel('Frequency')
            plt.ylabel('Normalized Power')
            plt.show()


    # Creating O-C with data
    def generate_oc(self) -> tuple[np.ndarray, np.ndarray]:
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
            
        plt.xlabel("Corrected Epoch")
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
        outsuffix: Suffix to append to output filenames.
        fitted_model: The updated model instance with fitted parameter values.
    """

    def __init__(self, model, data):
        """
        Initializes the fit instance.

        Parameters:
            model: The model instance (with a list of model_components).
            data: The observational data object used for fitting.
        """
        self.model = model
        self.data = data
        self.samples = None  
        self.outsuffix = None

    def model_params(self, model=None):
        dic = {}
        if model is None:
            if hasattr(self, "fitted_model"):
                model = self.fitted_model
            else:
                model = self.model
        for mc in model.model_components:
            for param in mc.params:
                #TODO aşağıdakini uncomment et 2 alttakini sil
                #dic[mc.name + " " +param] = mc.params[param]
                dic[param] = mc.params[param]
        for mc in model.model_components:
            print(mc.params)
        return dic

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
                coerce_farray=True, **kwargs):
        """
        Fits the model to the observational data using non-linear least-squares minimization
        with the total_oc_delay function, letting total_oc_delay handle unit conversions.
        """
        # Prepare variable parameter names in order.
        variable_param_names = []
        for component in self.model.model_components:
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
            return self.total_oc_delay(
                variable_params, m1=0, m2=0, inc=0, Ecorr=x, Mintimes=self.data.Mintimes, fix_units_first=True
            )

        # Assemble lmfit Parameters with **raw values**, no conversion.
        params = lm.Parameters()
        for component in self.model.model_components:
            identifier = component.name
            component.Ref_mintime = self.model.Ref_mintime
            component.Ref_period = self.model.Ref_period
            for attr, value in component.params.items():
                param_name = f'{attr}_{identifier}'
                param_value = getattr(component, attr).value
                param_min = getattr(component, attr).min
                param_max = getattr(component, attr).max
                params.add(param_name, value=param_value, vary=value.vary, 
                        min=param_min, max=param_max, expr=value.expr, 
                        brute_step=value.brute_step)

        # Create lmfit Model using the total_oc_delay wrapper.
        model = lm.Model(total_oc_delay_lmfit, independent_vars=['x'], nan_policy=self.model.nan_policy)

        weights = self.data.Weights
        if np.any(np.isnan(weights)):
            raise ValueError("Weights cannot contain NaN values. Fix nan values of your data weights.")

        # Perform the fit.
        result = model.fit(
            self.data.OC, params, x=self.data.Ecorr, weights=weights, 
            method=method, iter_cb=iter_cb, scale_covar=scale_covar, 
            fit_kws=fit_kws, nan_policy=nan_policy, calc_covar=calc_covar, 
            max_nfev=max_nfev, coerce_farray=coerce_farray, **kwargs
        )

        # Update the fitted model.
        self.fitted_model = self.create_fit_model(result)
        return result

    def fit_lin(self, inplace=True, plot=False):
        """
        Fits a simple linear model to the O-C data and removes the linear trend.

        Parameters:
            inplace (bool): If True, changes Ref_period and Ref_mintime using the fit, then recalculates epoch and O-C.
            plot (bool): If True, plots the original O-C data with the fitted linear trend.

        Returns:
            result: The lmfit result from the linear fit.
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
        Creates an updated model instance by setting each component's parameter to the fitted value.
        If stderr (uncertainty) is not available, sets it to np.nan.
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
        Creates a new model instance from MCMC samples by setting variable parameters
        to the median of the sample distribution.

        Parameters:
            samples (np.ndarray): Array of MCMC samples.

        Returns:
            new_model: A deepcopy of the original model with parameters updated to the sample medians.
        """
        p0 = []
        variable_indices = []

        # Collect initial parameter values and record indices for variable parameters.
        for model_component in self.model.model_components:
            for attr, value in model_component.params.items():
                p0.append(value.value)
                if value.vary:
                    variable_indices.append(len(p0) - 1)
                    
        s, ndim = samples.shape
        # Compute the 16th, 50th, and 84th percentiles.
        results_mcmc = list(map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                 zip(*np.percentile(samples, [16, 50, 84], axis=0))))
        results_par = [results_mcmc[i][0] for i in range(ndim)]
        results_std = np.std(samples, axis=0)
        counter0 = 0
        counter1 = 0
        new_model = copy.deepcopy(self.model)
        # Update each variable parameter with the median sample value.
        for model_component in new_model.model_components:
            for parameter in model_component.params:
                if counter1 in variable_indices:
                    setattr(model_component, parameter, results_par[counter0])
                    setattr(getattr(model_component, parameter), "std", results_std[counter0])
                    counter0 += 1
                counter1 += 1
        new_model.Ref_mintime = self.data.Ref_mintime
        new_model.Ref_period = self.data.Ref_period
        return new_model
        
    def read_samples(self, sample_file):
        """
        Reads MCMC samples from a text file.

        Parameters:
            sample_file (str): Path to the sample file.

        Returns:
            np.ndarray: Array of samples.
        """
        samples = np.loadtxt(sample_file)
        return samples

    def trace_plot(self, sampler, outsuffix="", save_plots=False, show=False):
        """
        Generates trace plots for each parameter from an emcee sampler with proper parameter names.

        Parameters:
            sampler: An emcee sampler object.
            outsuffix (str): Suffix for the output filename.
            save_plots (bool): If True, saves the plot to a file.
            show (bool): If True, displays the plot.

        Returns:
            None
        """
        import math

        nwalkers, nsteps, ndim = sampler.chain.shape
        ncols = 3  # Define the number of columns in the subplot grid.
        nrows = math.ceil(ndim / ncols)

        # Extract parameter labels
        labels = []
        for model_component in self.model.model_components:
            for parameter in model_component.params.keys():
                if model_component.params[parameter].vary:
                    labels.append(model_component.name + "_" + parameter)

        plt.figure(figsize=(14, nrows * 2.5))

        for h in range(ndim):
            plt.subplot(nrows, ncols, h + 1)
            plt.plot(sampler.chain[:, :, h].T, alpha=0.7)
            plt.title(labels[h], fontsize=12)
            plt.xlabel("Steps", fontsize=10)
            plt.ylabel("Value", fontsize=10)

        plt.tight_layout()

        if save_plots:
            plt.savefig(self.data.object_name + "_trace_" + str(outsuffix) + ".png")
        if show:
            plt.show()
        plt.close()


    def corner_plot(self, samples, outsuffix="", save_plots=False, show=False):
        """
        Generates a corner plot for the MCMC samples with median markers and cross lines.

        Parameters:
            samples (np.ndarray): Array of MCMC samples.
            outsuffix (str): Suffix for the output filename.
            save_plots (bool): If True, saves the plot to a file.
            show (bool): If True, displays the plot.

        Returns:
            None
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
            fig.savefig(self.data.object_name + "_corner_" + str(outsuffix) + ".png")
        plt.close()
        return fig

    def clear_emcee_sample(self, samples, threshold=0, order=0, clear_count=np.inf, inplace=False):
        """
        Filters out problematic samples from an emcee sample array using histogram-based criteria.

        Parameters:
            samples (np.ndarray): Raw sample array from emcee.
            threshold (float): Threshold multiplier for histogram bin counts.
            order (int): Order in which to process columns (0 for forward, 1 for reverse).
            clear_count (int): Maximum number of iterations for clearing.
            inplace (bool): Changes fitted_model with newly created samples when True.
            
        Returns:
            np.ndarray: The filtered sample array.
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
    
    def find_orbital_parameters(self, inc, inc_std, m1, m1_std, m2, m2_std):
        """
        Calculates orbital parameters (e.g., semi-major axis, mass function) using an external calculator.

        Parameters:
            inc (float): Inclination angle.
            inc_std (float): Standard deviation of the inclination.
            m1 (float): Primary mass.
            m1_std (float): Standard deviation of the primary mass.
            m2 (float): Companion mass.
            m2_std (float): Standard deviation of the companion mass.

        Returns:
            dict: A dictionary containing various calculated orbital parameters.
        """
        import orbit_param_calculator
        from uncertainties import ufloat
        if not hasattr(self, "fitted_model"):
            raise ValueError("No fitted model available. Please fit the model first.")
        dict_op = {}
        dp = 0
        dpstderr = 0
        self.fitted_model.Ref_period = self.data.Ref_period
        self.fitted_model.Ref_mintime = self.data.Ref_mintime
        
        for component in self.fitted_model.model_components:
            if hasattr(component, "dP"):
                dp = getattr(component, "dP").value
                dpstderr = getattr(component, "dP").std
            if hasattr(component, "T_LiTE"):
                # Convert parameters to the main units.
                P_LiTE_unit_fixed = _unit_conv(
                    getattr(component, "P_LiTE").value,
                    getattr(component, "P_LiTE").unit,
                    component._main_units["P_LiTE"],
                    ref_period=self.data.Ref_period, ref_mintime=self.data.Ref_mintime, parameter_name="P_LiTE",
                    JD_convertable=getattr(component, "P_LiTE").JD_convertable
                )
                P_LiTE_std_unit_fixed = _unit_conv(
                    getattr(component, "P_LiTE").std,
                    getattr(component, "P_LiTE").unit,
                    component._main_units["P_LiTE"],
                    ref_period=self.data.Ref_period, ref_mintime=self.data.Ref_mintime, parameter_name="P_LiTE",
                    JD_convertable=getattr(component, "P_LiTE").JD_convertable
                )
                amp_unit_fixed = _unit_conv(
                    getattr(component, "amp").value,
                    getattr(component, "amp").unit,
                    component._main_units["amp"],
                    ref_period=self.data.Ref_period, ref_mintime=self.data.Ref_mintime, parameter_name="amp",
                    JD_convertable=getattr(component, "amp").JD_convertable
                )
                amp_std_unit_fixed = _unit_conv(
                    getattr(component, "amp").std,
                    getattr(component, "amp").unit,
                    component._main_units["amp"],
                    ref_period=self.data.Ref_period, ref_mintime=self.data.Ref_mintime, parameter_name="amp",
                    JD_convertable=getattr(component, "amp").JD_convertable
                )
                omega_unit_fixed = _unit_conv(
                    getattr(component, "omega").value,
                    getattr(component, "omega").unit,
                    component._main_units["omega"],
                    ref_period=self.data.Ref_period, ref_mintime=self.data.Ref_mintime, parameter_name="omega",
                    JD_convertable=getattr(component, "omega").JD_convertable
                )
                omega_std_unit_fixed = _unit_conv(
                    getattr(component, "omega").std,
                    getattr(component, "omega").unit,
                    component._main_units["omega"],
                    ref_period=self.data.Ref_period, ref_mintime=self.data.Ref_mintime, parameter_name="omega",
                    JD_convertable=getattr(component, "omega").JD_convertable
                )
                
                # Calculate orbital parameters using the external orbit_param_calculator.
                period3_yr1 = orbit_param_calculator.period_in_years(
                    P_LiTE_unit_fixed, P_LiTE_std_unit_fixed, self.data.Ref_period + dp, dpstderr
                )
                sma12sini1 = orbit_param_calculator.sma12sini(
                    amp_unit_fixed, amp_std_unit_fixed, component.e.value, component.e.std, 
                    omega_unit_fixed, omega_std_unit_fixed
                )
                sma121 = orbit_param_calculator.sma12(
                    sma12sini1.nominal_value, sma12sini1.std_dev, inc, inc_std
                )
                mass_func1 = orbit_param_calculator.mass_func(
                    period3_yr1.nominal_value, period3_yr1.std_dev, 
                    sma12sini1.nominal_value, sma12sini1.std_dev
                )
                mass31sini = orbit_param_calculator.mass3(
                    mass_func1.nominal_value, mass_func1.std_dev, m1, m1_std, m2, m2_std
                )
                mass31 = orbit_param_calculator.mass3(
                    mass_func1.nominal_value, mass_func1.std_dev, m1, m1_std, m2, m2_std, inc, inc_std
                )
                mass31_mjup = mass31 * 1047.56
                sma121 = orbit_param_calculator.sma12(
                    sma12sini1.nominal_value, sma12sini1.std_dev, inc, inc_std
                )
                sma31 = orbit_param_calculator.sma3(
                    sma121.nominal_value, sma121.std_dev, m1, m1_std, m2, m2_std, 
                    mass31.nominal_value, mass31.std_dev
                )

                from uncertainties import ufloat
                P_LiTE = ufloat(component.P_LiTE.value, component.P_LiTE.std)
                Ref_period = ufloat(self.data.Ref_period, dpstderr)
                Ref_mintime = ufloat(self.data.Ref_mintime, 0)
                p_day = P_LiTE * Ref_period
                p_year = p_day / 365.242199
                T_LiTE = component.T_LiTE.value * Ref_period + Ref_mintime

                # Populate the dictionary with orbital parameters.
                dict_op[f"a12_sini{component.name}"] = sma12sini1
                dict_op[f"a12{component.name}"] = sma121
                dict_op[f"m_sini_msol{component.name}"] = mass31sini
                dict_op[f"m_sini_mjup{component.name}"] = mass31sini * 1047.56
                dict_op[f"m_msol{component.name}"] = mass31
                dict_op[f"m_jup{component.name}"] = mass31_mjup
                dict_op[f"a_AU{component.name}"] = sma31
                dict_op[f"p_day{component.name}"] = p_day
                dict_op[f"p_year{component.name}"] = p_year
                dict_op[f"ecc{component.name}"] = ufloat(component.e.value, component.e.std)
                dict_op[f"omega{component.name}"] = ufloat(component.omega.value, component.omega.std)
                dict_op[f"T_LiTE{component.name}"] = T_LiTE

        return dict_op

    def log_likelihood(self, variable_params, m1, m2, inc):
        observed_oc = self.data.OC
        observed_errors = self.data.Errors
        simulated_oc = self.total_oc_delay(variable_params, m1, m2, inc, Ecorr=None, Mintimes=None)
        residuals = observed_oc - simulated_oc
        return -0.5 * np.sum(((residuals)**2 / observed_errors**2) + np.log(observed_errors**2))

    def log_prior(self, variable_params):
        ln_sum = 0.0
        param_index = 0
        for component in self.model.model_components:
            for param_name, param_obj in component.params.items():
                if param_obj.vary:
                    param_value = variable_params[param_index]
                    if (param_obj.min is not None and param_value < param_obj.min) or \
                       (param_obj.max is not None and param_value > param_obj.max):
                        return -np.inf
                    ln_sum += fit.gaussian(param_value, param_obj.value, param_obj.std / 2.0)
                    param_index += 1
        if self.prob_prior:
            return ln_sum
        else:
            return 0

    def log_prob(self, variable_params, m1, m2, inc):
        lp = self.log_prior(variable_params)
        if not np.isfinite(lp):
            return -np.inf
        likelihood = self.log_likelihood(variable_params, m1, m2, inc)
        return lp + likelihood if np.isfinite(likelihood) else -np.inf      

    def total_oc_delay(self, variable_params, m1, m2, inc, Ecorr=None, Mintimes=None, fix_units_first=False):
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
        Mintimes = Mintimes if Mintimes is not None else self.data.Mintimes
        simulated_oc = 0
        param_index = 0
        lite_params = []
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
                simulated_oc += component.individual_model(Ecorr, *params)

        if lite_params:
            simulated_oc += self.simulate_oc_delay_lite(m1, m2, inc, lite_params, Mintimes)
        return simulated_oc

    def _extract_component_params(self, component, variable_params, start_index):
        params = []
        for param_name, param_obj in component.params.items():
            param_value = variable_params[start_index] if param_obj.vary else param_obj.value
            params.append(param_value)
            if param_obj.vary:
                start_index += 1
        return params, start_index

    def simulate_oc_delay_lite(self, m1, m2, system_inclination_deg, lite_params, mintimes):
        # Extract parameters, ensuring lite_params is a list of dictionaries
        times_of_p = [p.get('T_LiTE', Parameter(0)) for p in lite_params]
        periods = [p.get('P_LiTE', Parameter(0)) for p in lite_params]
        eccs = [p.get('ecc', Parameter(0)) for p in lite_params]
        omegas = [p.get('omega', Parameter(0)) for p in lite_params]
        masses = [p.get('mass', Parameter(0)) for p in lite_params]
        incs = [p.get('inc', Parameter(0)) for p in lite_params]
        return simulate_oc_delay(m1, m2, system_inclination_deg, times_of_p, periods, eccs, omegas, masses, incs, mintimes)

    @staticmethod
    def gaussian(p, mu, sigma):
        res = np.log(1.0 / (np.sqrt(2 * np.pi) * sigma)) - 0.5 * (p - mu)**2 / sigma**2
        if np.isnan(res):
            print("NAN", p, mu, sigma)
        return res


    def fit_model_prob(self, walker=20, steps=100, burn_in=0, threads=4, std_scale=0.05, 
                    create_sample_file=False, create_percentile_file=False, trace_plot=True, 
                    corner_plot=True, sample_plot=True, save_plots=False, show_plots=True, 
                    prob_prior=True, return_samples=True, return_sampler=False, multiprocessing=True):
        """
        Fits the model using MCMC sampling with the emcee package.

        This method sets up the initial positions, runs the sampler, and creates a new model from
        the posterior samples. It also saves sample and percentile files if requested and generates
        diagnostic plots.

        Returns:
            np.ndarray: The MCMC samples (as a flat array after burn-in) converted to the specified unit.
        """
        import copy
        import os
        import numpy as np
        import emcee
        from multiprocessing import Pool
        import matplotlib.pyplot as plt

        # Create a deepcopy of the model and update its units.
        fit2 = copy.deepcopy(self)
        fit2.model.Ref_period = fit2.data.Ref_period
        fit2.model.Ref_mintime = fit2.data.Ref_mintime
        fit2.model = fit2.model.fix_units()

        if burn_in > steps:
            raise ValueError("Burn-in value cannot be greater than the number of steps.")

        # Check for incompatible model components.
        lite_abspar_count = sum(isinstance(i, LiTE_abspar) for i in fit2.model.model_components)
        lite_count = sum(isinstance(i, LiTE) for i in fit2.model.model_components)
        if lite_abspar_count > 0 and lite_count > 0:
            raise ValueError("Cannot have both LiTE_abspar and LiTE components in the model.")
        if lite_abspar_count > 0:
            if fit2.data.m1 is None or fit2.data.m2 is None:
                raise ValueError("m1, m2, and inc must be provided for LiTE_abspar model.")
            m1, m2, inc = fit2.data.m1, fit2.data.m2, fit2.data.inc
        else:
            m1, m2, inc = 0, 0, 0

        fit2.prob_prior = prob_prior
        initial_params, initial_stds, min_bounds, max_bounds = [], [], [], []

        # Gather initial parameter values and bounds for all variable parameters.
        for component in fit2.model.model_components:
            for param_name, param_obj in component.params.items():
                if param_obj.vary:
                    initial_params.append(param_obj.value)
                    initial_stds.append((param_obj.std if param_obj.std is not None else 1e-4) * std_scale)
                    min_bounds.append(param_obj.min if param_obj.min is not None else -np.inf)
                    max_bounds.append(param_obj.max if param_obj.max is not None else np.inf)

        nwalkers = walker
        ndim = len(initial_params)

        # Initialize walker positions.
        initial_positions, nwalkers, ndim = fit2._initialize_sampling_params(walker, std_scale)

        # Run the MCMC sampler.
        if multiprocessing:
            with Pool(threads) as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, fit2.log_prob, args=(m1, m2, inc), pool=pool)
                sampler.run_mcmc(initial_positions, steps, progress=True)
        else:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, fit2.log_prob, args=(m1, m2, inc))
            sampler.run_mcmc(initial_positions, steps, progress=True)

        # Get the full sample chain.
        # Note: In recent versions of emcee, the shape is (nsteps, nwalkers, ndim)
        chains = sampler.get_chain(flat=False)
        print(f"Using all {chains.shape[1]} walkers without filtering for stuck walkers.")

        # Build a mapping of variable parameter names to their indices.
        variable_params = []
        for component in fit2.model.model_components:
            for param_name, param_obj in component.params.items():
                if param_obj.vary:
                    variable_params.append(param_name)
        param_indices = {param_name: idx for idx, param_name in enumerate(variable_params)}

        # Convert the chain to the specified unit.
        chains_conv = chains.copy()  # shape: (nsteps, nwalkers, ndim)
        for component in fit2.model.model_components:
            for param_name, param_obj in component.params.items():
                if param_obj.vary:
                    idx = param_indices[param_name]
                    chains_conv[:, :, idx] = _unit_conv(
                        chains_conv[:, :, idx],
                        component._main_units[param_name],
                        param_obj.unit,
                        ref_period=fit2.data.Ref_period,
                        ref_mintime=fit2.data.Ref_mintime,
                        parameter_name=param_name,
                        JD_convertable=param_obj.JD_convertable
                    )

        # For plotting purposes, flatten the entire converted chain.
        samples_full = chains_conv.reshape(-1, ndim)
        samples_old = copy.deepcopy(samples_full)

        # Apply burn-in correctly: remove the first `burn_in` steps (axis 0) then flatten.
        samples = chains_conv[burn_in:, :, :].reshape(-1, ndim)
        if samples.size == 0:
            raise ValueError("No samples remain after applying burn_in. "
                            "Please check that the burn_in parameter is less than the number of steps.")
        print("samples burned in")

        # Update the fitted model based on the sampled parameters.
        self.fitted_model = self.create_model_from_samples(samples)
        print("new model created")

        # Build a unique filename suffix for outputs.
        identifier = "".join([component.name for component in fit2.model.model_components])
        outsuffix = f"{nwalkers}_{steps}_{identifier}"
        base_filename = f"{fit2.data.object_name}_prios_{outsuffix}.out"
        filename = base_filename
        counter = 1
        while os.path.exists(filename):
            filename = f"{base_filename}_{counter}.out"
            counter += 1
        outsuffix = f"{nwalkers}_{steps}_{identifier}_{counter}"

        if create_sample_file:
            sample_filename = f"{fit2.data.object_name}_emcee_samples_{outsuffix}.out"
            print(f"Saving MCMC samples to {sample_filename}...")
            np.savetxt(sample_filename, samples, delimiter=" ", fmt="%.8e")

        results_mcmc = list(map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                zip(*np.percentile(samples, [16, 50, 84], axis=0))))

        if create_percentile_file:
            percentile_filename = f"{fit2.data.object_name}_emcee_percentiles_{outsuffix}.out"
            print(f"Saving percentile results to {percentile_filename}...")
            np.savetxt(percentile_filename, results_mcmc, delimiter=" ", fmt="%.8e")

        # Generate diagnostic plots if requested.
        if trace_plot:
            self.trace_plot(sampler, outsuffix="filtered", save_plots=save_plots, show=show_plots)
        if corner_plot:
            self.corner_plot(samples, outsuffix="filtered", save_plots=save_plots, show=show_plots)
        if sample_plot:
            self.plot(samples=samples, show=True)

        plt.show()

        if return_sampler and not return_samples:
            return sampler
        if return_samples and not return_sampler:
            return samples
        else:
            return samples, sampler


    def _initialize_sampling_params(self, walker, std_scale):
        """
        Prepares initial positions for the MCMC sampler.

        Parameters:
            walker (int): Number of walkers.
            std_scale (float): Scale factor for the standard deviation of initial positions.

        Returns:
            tuple: (initial_positions, nwalkers, ndim)
        """
        initial_params, initial_stds, min_bounds, max_bounds = [], [], [], []
        for component in self.model.model_components:
            for param_obj in component.params.values():
                if param_obj.vary:
                    initial_params.append(param_obj.value)
                    initial_stds.append((param_obj.std or 1e-4) * std_scale)
                    min_bounds.append(param_obj.min or -np.inf)
                    max_bounds.append(param_obj.max or np.inf)
        
        nwalkers, ndim = walker, len(initial_params)
        initial_positions = np.clip(
            np.random.normal(initial_params, initial_stds, (nwalkers, ndim)),
            min_bounds, max_bounds
        )
        return initial_positions, nwalkers, ndim
            
    def plot_orbit_gif(self, output_file="orbit.gif", time_count=1000, real_inc=False):
        """
        Plots the system's orbits and creates a GIF animation with an accompanying light-time effect graph.

        Parameters:
            output_file (str): Filename for the output GIF.
            time_count (int): Number of time steps for the animation.
            real_inc (bool): If True, uses real inclinations for orbit animation; otherwise uses adjusted values.

        Returns:
            None
        """
        fit2 = copy.deepcopy(self)
        fit2.model.Ref_period = fit2.data.Ref_period
        fit2.model.Ref_mintime = fit2.data.Ref_mintime
        fit2.model = fit2.model.fix_units()
        
        # Extract system parameters.
        m1 = fit2.data.m1
        m2 = fit2.data.m2
        system_inclination_deg = fit2.data.inc

        # Extract orbital parameters from LiTE_abspar components.
        times_of_p = []
        periods = []
        eccentricities = []
        omegas_deg = []
        masses = []
        incs_real = []  # Real inclinations.
        incs_used = []  # Inclinations for animation.
        for component in fit2.model.model_components:
            if isinstance(component, LiTE_abspar):
                times_of_p.append(component.params["T_LiTE"].value)
                periods.append(component.params["P_LiTE"].value)
                eccentricities.append(component.params["ecc"].value)
                omegas_deg.append(component.params["omega"].value)
                masses.append(component.params["mass"].value)
                incs_real.append(component.params["inc"].value)
                incs_used.append(component.params["inc"].value if real_inc else 0)

        # Create a new mintimes array for smooth animation.
        original_mintimes = fit2.data.Mintimes
        mintimes = np.linspace(min(original_mintimes), max(original_mintimes), time_count)

        # Prepare a copy of the current fit instance for O-C calculation.
        fit2 = copy.deepcopy(self)
        fit2.data.df = pd.DataFrame()
        fit2.data.Mintimes = mintimes
        fit2.data.Ecorr = np.linspace(min(fit2.data.Ecorr), max(fit2.data.Ecorr), time_count)
        fit2.model.epochs = fit2.data.Ecorr

        # Freeze parameters for the simulation.
        for model_component in fit2.model.model_components:
            for param in model_component.params.values():
                param.vary = False

        variable_params = []
        total_oc_delay = fit2.total_oc_delay(variable_params, m1, m2, system_inclination_deg)
        plt.plot(fit2.data.Ecorr, total_oc_delay)

        # Calculate positions for animation using simulate_oc_delay.
        _, positions = simulate_oc_delay(
            m1=m1,
            m2=m2,
            system_inclination_deg=0 if not real_inc else system_inclination_deg,
            times_of_p=times_of_p,
            periods=periods,
            eccentricities=eccentricities,
            omegas_deg=omegas_deg,
            masses=masses,
            incs=incs_used,
            mintimes=mintimes,
            gif=True
        )

        # Set up the figure with two subplots: one for the orbit, one for the light-time effect.
        fig, (ax_orbit, ax_oc) = plt.subplots(2, 1, figsize=(6, 10))
        fig.subplots_adjust(hspace=0.4)

        # Configure orbit subplot.
        max_radius = max(
            max(np.linalg.norm(np.array(pos), axis=1)) for pos in positions.values()
        )
        ax_orbit.set_xlim(-max_radius - 1, max_radius + 1)
        ax_orbit.set_ylim(-max_radius - 1, max_radius + 1)
        ax_orbit.set_xlabel("X (AU)")
        ax_orbit.set_ylabel("Y (AU)")
        ax_orbit.grid(True)
        lines_orbit = [ax_orbit.plot([], [], 'o', label=f"Body {i}")[0] for i in range(len(positions))]
        ax_orbit.legend()

        # Configure light-time effect subplot.
        ax_oc.set_xlim(min(fit2.data.Ecorr), max(fit2.data.Ecorr))
        ax_oc.set_xlabel("Epoch")
        ax_oc.set_ylabel("Total O-C Delay (Days)")
        ax_oc.grid(True)
        ax_oc.plot(fit2.data.Ecorr, fit2.data.OC, "r.", label="Observed O-C")
        line_total_oc, = ax_oc.plot([], [], label="Total O-C", color="black", linestyle="dashed")
        ax_oc.legend()

        # Define the update function for the animation.
        def update(frame):
            # Update orbit positions.
            for i, line in enumerate(lines_orbit):
                x, y = positions[i][frame]
                line.set_data([x], [y])
            # Update the O-C delay plot.
            line_total_oc.set_data(fit2.data.Ecorr[:frame + 1], total_oc_delay[:frame + 1])
            return lines_orbit + [line_total_oc]

        ani = animation.FuncAnimation(
            fig, update, frames=time_count, interval=100, blit=True
        )
        ani.save(output_file, writer="pillow")
        plt.close(fig)
        print(f"Orbit animation saved as {output_file}")
        
    def sample_plot(self, samples, nrandom_samples=100, outsuffix="", save_plots=False, show=False, create_median_file=False):
        """
        Plots the data groups and the sampled models.

        Args:
            samples (numpy.ndarray): Array of MCMC samples.
            nrandom_samples (int): Number of random samples to plot from the posterior.
            outsuffix (str): Suffix for saving the file.
            save_plots (bool): Whether to save the plots.
            show (bool): Whether to display the plots.
            create_median_file (bool): Whether to save the median file.

        Returns:
            None
        """
        Mintimes = self.data.Mintimes
        Ecorr = self.data.Ecorr 
        daystosec = 24 * 60 * 60
        p0 = []
        variable_indices = []
        e_th = np.linspace(min(self.data.Ecorr), max(self.data.Ecorr), 3000)

        # Collect parameter values and bounds
        for model_component in self.model.model_components:
            for attr, value in model_component.params.items():
                p0.append(value.value)
                if value.vary:
                    variable_indices.append(len(p0) - 1)

        def function(*args):
            return self.total_oc_delay(args, self.data.m1, self.data.m2, self.data.inc, Ecorr, Mintimes)

        # Use a colormap for different data groups
        unique_groups = np.unique(self.data.Data_group)  # Get unique data groups
        colors = plt.get_cmap('tab10', len(unique_groups))
        color_dict = {name: colors(i) for i, name in enumerate(unique_groups)}

        # Plot samples by group
        grouped = pd.DataFrame(self.data.df).groupby("Data_group")  # Ensure it's a DataFrame for grouping
        plt.figure(figsize=(10, 6))
        for name, group in grouped:
            plt.plot(group["Ecorr"], group["OC"]*daystosec, ".", label=name, color=color_dict[name])

        # Randomly sample the posterior samples for plotting
        indices = np.random.choice(range(len(samples)), size=nrandom_samples, replace=False)
        for idx in indices:
            sample = samples[idx]
            y_fit = function(*sample)
            plt.plot(e_th, y_fit, alpha=0.1, color='grey')  # Light grey for sample fits

        plt.xlabel("Corrected Epoch", fontsize=16)
        plt.ylabel("O-C (Sec)", fontsize=16)
        plt.legend(title="Data Group")
        
        s, ndim = samples.shape
        results_mcmc = list(map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0))))
        results_par = [results_mcmc[i][0] for i in range(ndim)]
        symmetric_errors = [(results_mcmc[i][1] + results_mcmc[i][2]) / 2 for i in range(ndim)]
        par_percent = []
        for n in range(3):
            for b in range(ndim):
                par_percent.append(results_mcmc[b][n])
        midpar, pluspar, minuspar = par_percent[:ndim], par_percent[ndim:ndim*2], par_percent[ndim*2:]
        model_err_p, model_err_m = self.error_propagate(function, midpar, pluspar, minuspar)
        
        for i, params in enumerate(samples[np.random.randint(len(samples), size=nrandom_samples)]):
            break
        plt.fill_between(e_th,fit.model(e_th, params, self.model.model_components, variable_indices)*daystosec+3*model_err_p*daystosec,\
			fit.model(e_th, params, self.model.model_components, variable_indices)*daystosec+3*model_err_m*daystosec,color="gray",alpha=0.1, label="Fit error")
        # Plot the model with uncertainty
        plt.plot(e_th, fit.model(e_th, results_par, self.model.model_components, variable_indices) * daystosec, color="r", label="Median fit")

        if save_plots:
            plt.savefig(f"{self.data.object_name}_sample_plot_{outsuffix}.png")
        if show:
            plt.show()
        plt.close()
        
    @staticmethod
    def error_propagate(function, args, std_arg_p, std_arg_m=False, std_m_negative=False):
        """
        Propagates uncertainties through a given function using finite difference approximations.

        Parameters:
            function (callable): The function for which to propagate error.
            args (list or array): The nominal values of the arguments.
            std_arg_p (list or array): The positive standard deviations for each argument.
            std_arg_m (list or bool): The negative standard deviations or False.
            std_m_negative (bool): If True, negates std_arg_m.

        Returns:
            If std_arg_m provided: tuple (error_plus, error_minus)
            Otherwise: error_plus.
        """
        import numpy as np
        std_args_p = np.asarray(std_arg_p)
        alpha_Z_p_sqr = 0.

        if not isinstance(std_arg_m, bool):
            std_args_m = np.asarray(std_arg_m)
            if std_m_negative:
                std_args_m = (-1) * std_args_m
            alpha_Z_m_sqr = 0.

        for i in range(len(args)):
            new_args = np.array(args)
            new_args[i] = args[i] + std_args_p[i]
            alpha_p = function(*new_args) - function(*args)
            alpha_Z_p_sqr += alpha_p**2
            if not isinstance(std_arg_m, bool):
                new_args = np.array(args)
                new_args[i] = args[i] - std_args_m[i]
                alpha_m = function(*new_args) - function(*args)
                alpha_Z_m_sqr += alpha_m**2

        alpha_Z_p = np.sqrt(alpha_Z_p_sqr)
        if not isinstance(std_arg_m, bool):
            alpha_Z_m = np.sqrt(alpha_Z_m_sqr)
            return alpha_Z_p, alpha_Z_m * (-1)
        return alpha_Z_p

    def calculate_fit_uncertanity(self, samples, Ecorr=None, Mintimes=None):
        """
        Calculates uncertainties in the fitted model's O-C delay using MCMC samples.

        Parameters:
            samples (np.ndarray): Array of MCMC samples in the original (non-fixed) units.
            Ecorr (np.ndarray): Optional array of epochs (defaults to self.data.Ecorr).
            Mintimes (np.ndarray): Optional array of mintimes (defaults to self.data.Mintimes).

        Returns:
            tuple: (positive uncertainty, negative uncertainty)
        """
        fit2 = copy.deepcopy(self)
        fit2.model.Ref_period = fit2.data.Ref_period
        fit2.model.Ref_mintime = fit2.data.Ref_mintime
        fit2.model = fit2.model.fix_units()
        
        Mintimes = Mintimes if Mintimes is not None else fit2.data.Mintimes
        Ecorr = Ecorr if Ecorr is not None else fit2.data.Ecorr 
        
        # Convert samples back to the original units
        param_indices = {param_name: idx for idx, param_name in enumerate([p for c in fit2.model.model_components for p in c.params if c.params[p].vary])}
        samples2 = copy.deepcopy(samples)
        for component in fit2.model.model_components:
            for param_name, param_obj in component.params.items():
                if param_obj.vary:
                    idx = param_indices[param_name]  # Get the correct index
                    samples2[:, idx] = _unit_conv(
                        samples2[:, idx], param_obj.unit, component._main_units[param_name], 
                        ref_period=fit2.data.Ref_period, ref_mintime=fit2.data.Ref_mintime,
                        parameter_name=param_name, JD_convertable=param_obj.JD_convertable
                    )
        
        s, ndim = samples2.shape
        perc = np.percentile(samples2, [16, 50, 84], axis=0)
        results_mcmc = list(map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]), zip(*perc)))
        midpar = [results_mcmc[i][0] for i in range(ndim)]
        pluspar = [results_mcmc[i][1] for i in range(ndim)]
        minuspar = [results_mcmc[i][2] for i in range(ndim)]
        
        def function(*args):
            return fit2.total_oc_delay(args, fit2.data.m1, fit2.data.m2, fit2.data.inc, Ecorr, Mintimes)
        
        model_err_p, model_err_m = fit2.error_propagate(function, midpar, pluspar, minuspar)
        print(model_err_p)
        
        return model_err_p, model_err_m


    def plot(self, **kwargs):
        """
        Plots the O-C data along with the fitted model.

        Parameters:
            samples (np.ndarray): MCMC samples to use for plotting.
            **kwargs: Additional keyword arguments passed to the sample_plot_nonclass function.

        Returns:
            The matplotlib Figure object produced by sample_plot_nonclass.
        """
        model = self.fitted_model if hasattr(self, "fitted_model") else self.model
        return sample_plot_nonclass(
            model=model,
            data=self.data,
            **kwargs
        )
    
    @staticmethod
    def _gaussian(p, mu, sigma):
        res = np.log(1.0 / (np.sqrt(2 * np.pi) * sigma)) - 0.5 * (p - mu)**2 / sigma**2
        if np.isnan(res):
            print("NAN", p, mu, sigma)
        return res


    
class Parameter:  
    
    """
    Represents a parameter with a value and a unit.

    Attributes:
        value: The numerical value of the parameter.
        unit: The unit of measurement for the parameter value.
    """ 
    def __init__(self, value, unit=None, min=None, max=None, vary=True, JD_convertable=False, expr=None, brute_step=None, user_data=None, std=0):
        """
        Initializes a Parameter object.

        Args:
            value: The numerical value of the parameter.
            unit: The unit of measurement for the parameter value.
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
    Parent class for all model components (Lin, Quad, LiTE).
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
        Set multiple parameters using a dictionary of {variable: value} pairs.

        Args:
            params (dict): Dictionary containing parameters and their values.

        Returns:
            None
        """
        for variable, value in params.items():
            setattr(self, variable, self._set_single_param(variable, value))
            if variable.startswith("T_"):
                getattr(self, variable).JD_convertable = True

    def _set_single_param(self, variable, value):
        """
        Set a single parameter value, either as a Parameter object or as a float with the appropriate unit.

        Args:
            variable (str): Name of the parameter.
            value: Value of the parameter.

        Returns:
            Parameter: Parameter object representing the parameter value.
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
        Plot the output of individual_model against the provided epochs.

        Args:
            epochs (array-like): Array-like object containing epochs.

        Returns:
            None
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
    Represents a linear model with parameters for temperature difference (dT), pressure difference (dP), and an optional name.
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
        Calculates the value of the linear function for a given input x.

        Parameters:
        - x (numeric): The input value for the linear function.
        - dT (float, optional): Temperature difference.
        - dP (float, optional): Pressure difference.

        Returns:
        - The result of the linear function calculation.
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
    Represents a quadratic model with a parameter for the quadratic coefficient (Q) and an optional name.
    """
    def __init__(self, params = {}, name="Quad", units={}):
        """
        Initializes a new instance of the Quad class with a specified quadratic coefficient and a name.

        Parameters:
        - params (dict): Dictionary containing parameter Q.
        - name (str): The name of the instance. Default is "Quad".
        """
        self._main_units = {"Q": "Unitless"}
        self.params = {"Q":0} if params == {} else params
        self.name = name
        self.Ref_period = None
        self.set_params(params=params)


    def individual_model(self, x, Q=None):
        """
        Calculates the value of the quadratic function for a given input x.

        Parameters:
        - x (numeric): The input value for the quadratic function.
        - Q (float, optional): Quadratic coefficient.

        Returns:
        - The result of the quadratic function calculation.
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
    Represents a model for calculating the Light-Time Effect (LiTE) in binary star systems or planetary systems.
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
        Calculates the LiTE effect for a given input x (time).

        Parameters:
        - x (numeric): The input time for which to calculate the LiTE effect.
        - e (float, optional): Eccentricity.
        - omega (float, optional): Argument of periapsis in degrees.
        - P_LiTE (float, optional): Orbital period in epochs.
        - T_LiTE (float, optional): Time of periastron passage in epochs.
        - amp (float, optional): Amplitude in days.

        Returns:
        - The result of the LiTE function calculation.
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
        
        # print("e=", parameter["e"])
        # print("omega=", parameter["omega"])
        # print("P_LiTE=", parameter["P_LiTE"])
        # print("T_LiTE=", parameter["T_LiTE"])
        # print("amp=", parameter["amp"])
        if isinstance(parameter["e"], float) and (parameter["e"] >= .999 or parameter["e"] < 0):
            return -1e90
        # print(parameter["e"])
        true_anom = Functions._anomaly(x, parameter["T_LiTE"], parameter["P_LiTE"], parameter["e"])
        LiTE = Functions._lite_formula(parameter["e"], parameter["amp"], parameter["omega"], true_anom)
        return LiTE
    
class LiTE_abspar(model_component):
    def __init__(self, params = {}, name="LiTE_abspar"):
        self._main_units = {"mass": "GM_sun", "P_LiTE":"day", "ecc": "Unitless", "omega":"deg", "T_LiTE":"BJD", "inc":"deg"}
        self.params = {"mass":0, "P_LiTE":0, "ecc":0, "omega":0, "T_LiTE":0, "inc":90} if params == {} else params
        self.name = name
        self.Ref_period = None
        self.set_params(params=params)
        
class Grav_rad(model_component):
    def __init__(self, params = {}, name="Grav"):
        self._main_units = {"a_grav": "day", "b_grav":"day", "c_grav": "day"}
        self.params = {"a_grav":0, "b_grav":0, "c_grav":0} if params == {} else params
        self.name = name
        self.Ref_period = None
        self.set_params(params=params)

    def individual_model(self, x, a_grav=None, b_grav=None, c_grav=None, Ref_period=None, Ref_mintime=None):
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
    def __init__(self, params = {}, name="Mag"):
        self._main_units = {"P_mag": "day", "A_mag":"day", "phi": "day", "c": "day"}
        self.params = {"P_mag":0, "A_mag":0, "phi":0, "c":0} if params == {} else params
        self.name = name
        self.Ref_period = None
        self.set_params(params=params)

    def individual_model(self, x, P_mag=None, A_mag=None, phi=None, c=None, Ref_period=None, Ref_mintime=None):
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
    Calculate AIC and BIC values.

    :param lnlike: Log-likelihood value at the best-fit parameters
    :param theta_best: Best-fit parameters (median of MCMC samples)
    :param k: Number of parameters
    :param n: Number of data points
    :return: AIC and BIC values
    """
    model = copy.deepcopy(model)
    data = copy.deepcopy(data)
    model.epochs = data.Ecorr
    param_count = 0
    for model_component in model.model_components:
        for param in model_component.params.values():
            if param.vary:
                param_count += 1
    chi2 = np.sum((model.calculate_oc() - data.OC) ** 2 / data.Errors ** 2)
    reduced_chi2 = chi2 / (len(data.OC) - param_count)
    AIC = chi2 + (2 * param_count)
    BIC = chi2 + (np.log(len(data.OC)) * param_count)
    # Print metrics for the current model
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
    def _anomaly(x, Ts,Per,ecc):
        x = np.array(x)
        mean_anom = (((x - Ts) / Per) * (2 * np.pi)) % (2 * np.pi)  # mean functions.anomaly in radian
        ecc_anom = Functions._kepler_solve(mean_anom, ecc)
        true_anom = 2 * np.arctan(np.sqrt((1 + ecc) / (1 - ecc)) * np.tan(ecc_anom / 2.))
        return true_anom

    @staticmethod
    def _lite_formula(e,amp,omega, true_anom):
        radians = np.radians
        LiTE = ((amp/(np.sqrt(1-(e**2)*(np.cos(radians(omega)))**2)))* \
            ((((1-e**2)/(1+e*np.cos(true_anom)))* \
            np.sin(true_anom+radians(omega)))+(e* \
            np.sin(radians(omega)))))
        return LiTE
    
    @staticmethod
    def _kepler_solve(m_anom, eccentricity):
        """Solves Kepler's equation for the eccentric anomaly
        using Newton's method.

        Arguments:
        m_anom -- mean anomaly in radians (can be array)
        eccentricity -- eccentricity of the orbit (should be 0 <= e <= 1)

        Returns: eccentric anomaly in radians.
        """

        # Check for invalid eccentricity values
        if np.any(eccentricity > 1) or np.any(eccentricity < 0):
            print("Error: Eccentricity must be in the range 0 <= e <= 1.")
            return np.full_like(m_anom, np.nan)

        desired_accuracy = 1e-5
        e_anom = np.array(m_anom, dtype=np.float64)  # Ensure floating-point precision
        counter = 0
        max_iterations = 10000  # Avoid infinite loops

        while True:
            counter += 1

            # Compute the difference in Newton's method
            diff = e_anom - (eccentricity * np.sin(e_anom)) - m_anom
            
            # Check for NaN or Inf in the calculation
            if np.any(np.isnan(diff)) or np.any(np.isinf(diff)):
                print("Warning: NaN or Inf encountered in diff calculation!")
                print(f"e_anom: {e_anom}")
                print(f"eccentricity: {eccentricity}")
                print(f"m_anom: {m_anom}")
                return np.full_like(m_anom, np.nan)

            denominator = 1 - eccentricity * np.cos(e_anom)
            
            # Avoid division by zero or extremely small numbers
            if np.any(np.abs(denominator) < 1e-10):
                return np.full_like(m_anom, np.nan)

            e_anom -= diff / denominator

            # Convergence check
            if np.all(np.abs(diff) <= desired_accuracy):
                break

            # Stop if iteration count is exceeded
            if counter > max_iterations:
                return np.full_like(m_anom, np.nan)

        return e_anom
    
def _unit_conv(value, from_unit, to_unit, ref_period=None, ref_mintime=None, parameter_name=None, JD_convertable=False):
    if from_unit == to_unit:
        return value
    elif to_unit != "epoch" and from_unit != "epoch":
        return _unit_conv_nonepoch(value, from_unit, to_unit)
    elif (to_unit == "epoch") and ref_period is None:
        raise ValueError(f"Ref_period is required for converting the unit named '{parameter_name}' to epoch.")
    elif (from_unit == "epoch") and ref_period is None:
        raise ValueError(f"Ref_period is required for converting the unit named '{parameter_name}' from epoch.")
    elif JD_convertable or from_unit=="BJD" or to_unit=="BJD" or from_unit=="HJD" or to_unit=="HJD":
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
    
def sample_plot_nonclass(model=None,
                         data=None,
                         samples=None,
                         nrandom_samples=100, 
                         outsuffix="", 
                         save_plots=False, 
                         show=False, 
                         create_median_file=False, 
                         other_models=None, 
                         color_palette="tab20", 
                         group_colors=None,
                         group_shapes=None,
                         group_sizes=None,
                         x_axis_bot="epoch", 
                         x_axis_top=None,
                         y_axis_left="second", 
                         y_axis_right=None,
                         extend_graph_factor=0.05,
                         graph_size=(14, 8),
                         label_size=16,
                         # Tick size parameters:
                         tick_size=12,
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
                         res_hspace=0.4):
    """
    Plots an O-C (Observed minus Calculated) diagram with error bars, a median fit,
    and optionally a residuals subplot.

    This version sets the top x-axis ticks so that:
      - The first tick is at January 1 of the year: floor(minimum_year) + 1.
      - The last tick is initially set to ceil(maximum_year).
      - We then check if (last_tick - first_tick) is divisible by 8; if not, try 7 then 6.
      - If none of these divisors work, we decrement last_tick until one works.
      - Tick labels are drawn at January 1 of the chosen years.
      
    (Depending on which divisor works, you may see 9, 8, or 7 tick labels.)
    The main x-axis limits remain tied to the data (or to x_lim) so that the graph width is not extended.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import copy
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator, FixedLocator, FixedFormatter

    # Set tick sizes (use tick_size if not provided)
    tick_size_x_bottom = tick_size_x_bottom or tick_size
    tick_size_x_top    = tick_size_x_top or tick_size
    tick_size_y_left   = tick_size_y_left or tick_size
    tick_size_y_right  = tick_size_y_right or tick_size

    def dynamic_format(value_range):
        if value_range <= 1e-5:
            return "%.7f"
        elif value_range <= 1e-4:
            return "%.6f"
        elif value_range <= 1e-3:
            return "%.5f"
        elif value_range <= 1e-2:
            return "%.4f"
        elif value_range <= 0.1:
            return "%.3f"
        elif value_range <= 1:
            return "%.2f"
        elif value_range <= 100:
            return "%.1f"
        else:
            return "%.0f"

    def approximate_year_from_bjd(bjd):
        # Year 2000 corresponds to BJD 2451545.0
        return 2000.0 + (bjd - 2451545.0) / 365.25

    def approximate_bjd_from_year(year):
        # Returns BJD corresponding to January 1 of the given year (approximate)
        return 2451545.0 + (year - 2000.0) * 365.25

    def _apply_bottom_xaxis(ax, x_axis_type):
        ax.tick_params(axis='x', labelsize=tick_size_x_bottom)
        if x_axis_type.lower() == "bjd":
            ref_mintime = getattr(data, "Ref_mintime", 0)
            ref_period  = getattr(data, "Ref_period", 1)
            bottom_ticks = ax.get_xticks()
            actual_bjd = [ref_mintime + (tick * ref_period) for tick in bottom_ticks]
            differences = [b - bjd_offset for b in actual_bjd]
            ax.xaxis.set_major_locator(FixedLocator(bottom_ticks))
            ax.xaxis.set_major_formatter(FixedFormatter([f"{d:.0f}" for d in differences]))
            ax.set_xlabel(f"BJD - {int(bjd_offset)}", fontsize=label_size)
        elif x_axis_type.lower() == "epoch":
            ax.set_xlabel("Cycle", fontsize=label_size)
        elif x_axis_type.lower() == "year":
            ax.set_xlabel("Year", fontsize=label_size)
        else:
            ax.set_xlabel(x_axis_type, fontsize=label_size)

    # Unit conversion factors
    conv_factors = {"day": 1, "hour": 24, "minute": 1440, "second": 86400, "millisecond": 86400000}
    factor_left = conv_factors[y_axis_left.lower()]
    factor_right = conv_factors[y_axis_right.lower()] if y_axis_right else None

    # Extend Ecorr for plotting the model:
    e_range_val = max(data.Ecorr) - min(data.Ecorr)
    egap = e_range_val * extend_graph_factor
    e_th_min = min(data.Ecorr) - egap
    e_th_max = max(data.Ecorr) + egap
    e_th = np.linspace(e_th_min, e_th_max, 3000)

    # Extend Mintimes for plotting the model:
    mintime_range = max(data.Mintimes) - min(data.Mintimes)
    mintime_gap = mintime_range * extend_graph_factor
    mintime_min = min(data.Mintimes) - mintime_gap
    mintime_max = max(data.Mintimes) + mintime_gap
    new_Mintimes = np.linspace(mintime_min, mintime_max, 3000)

    p0 = []
    variable_indices = []
    for comp in model.model_components:
        for _, val in comp.params.items():
            p0.append(val.value)
            if val.vary:
                variable_indices.append(len(p0) - 1)

    oc_days = data.df["OC"].values
    err_days = data.df["Errors"].values
    oc_left = oc_days * factor_left
    err_left = err_days * factor_left

    if y_axis_right:
        oc_right = oc_days * factor_right
    else:
        oc_right = None

    # Create figure (with or without residuals subplot)
    if res_plot:
        fig, (ax_main, ax_res) = plt.subplots(
            2, 1, sharex=True,
            gridspec_kw={'height_ratios': res_height_ratios, 'hspace': res_hspace},
            figsize=graph_size
        )
        ax_left = ax_main
    else:
        fig, ax_left = plt.subplots(figsize=graph_size)

    # Group data by "Data_group"
    df_reset = data.df.reset_index(drop=True)
    groups_list = list(df_reset.groupby("Data_group"))
    num_groups = len(groups_list)
    
    # Process group colors
    if group_colors is None:
        cmap = plt.get_cmap(color_palette, num_groups)
        group_color_list = [cmap(i) for i in range(num_groups)]
    else:
        if not isinstance(group_colors, list):
            group_color_list = [group_colors] * num_groups
        else:
            group_color_list = group_colors.copy()
        if len(group_color_list) < num_groups:
            group_color_list += [None] * (num_groups - len(group_color_list))
        cmap = plt.get_cmap(color_palette, num_groups)
        for i in range(num_groups):
            if group_color_list[i] is None:
                group_color_list[i] = cmap(i)
    
    # Process group shapes
    if group_shapes is None:
        group_shapes_list = ["."] * num_groups
    else:
        if not isinstance(group_shapes, list):
            group_shapes_list = [group_shapes] * num_groups
        else:
            group_shapes_list = group_shapes.copy()
        if len(group_shapes_list) < num_groups:
            group_shapes_list += [None] * (num_groups - len(group_shapes_list))
        group_shapes_list = [m if m is not None else "." for m in group_shapes_list]
    
    # Process group sizes
    if group_sizes is None:
        group_sizes_list = [5] * num_groups
    else:
        if not isinstance(group_sizes, list):
            group_sizes_list = [group_sizes] * num_groups
        else:
            group_sizes_list = group_sizes.copy()
        if len(group_sizes_list) < num_groups:
            group_sizes_list += [None] * (num_groups - len(group_sizes_list))
        group_sizes_list = [s if s is not None else 5 for s in group_sizes_list]
    
    group_color_dict = {}
    group_shape_dict = {}
    group_size_dict = {}
    for i, (name, grp) in enumerate(groups_list):
        group_color_dict[name] = group_color_list[i]
        group_shape_dict[name] = group_shapes_list[i]
        group_size_dict[name] = group_sizes_list[i]
        markerfacecolor = 'none' if group_shapes_list[i] == "o" else group_color_list[i]
        ax_left.errorbar(
            grp["Ecorr"],
            grp["OC"] * factor_left,
            yerr=grp["Errors"] * factor_left,
            fmt=group_shapes_list[i],
            markersize=group_sizes_list[i],
            mfc=markerfacecolor,
            alpha=1,
            elinewidth=1.2,
            capsize=2,
            color=group_color_list[i],
            label=name
        )
        
    # Perform the fit and compute the best-fit curve
    fit2 = fit(model, data)
    fit2 = copy.deepcopy(fit2)
    fit2.model.Ref_period = fit2.data.Ref_period
    fit2.model.Ref_mintime = fit2.data.Ref_mintime
    fit2.fitted_model = fit2.model.fix_units()
    fit2.model = fit2.model.fix_units()

    if samples is not None:
        model_err_p, model_err_m = fit2.calculate_fit_uncertanity(samples, Ecorr=e_th, Mintimes=new_Mintimes)
        for comp in fit2.model.model_components:
            for val in comp.params.values():
                val.vary = False
        best_fit = fit2.total_oc_delay([], fit2.data.m1, fit2.data.m2,
                                       fit2.data.inc, e_th, new_Mintimes)
        best_fit_left = best_fit * factor_left
        err_p_left = model_err_p * factor_left
        err_m_left = model_err_m * factor_left
        ax_left.fill_between(e_th, best_fit_left + err_p_left,
                             best_fit_left + err_m_left, color="gray", alpha=0.15)
    else:
        for comp in fit2.model.model_components:
            for val in comp.params.values():
                val.vary = False
        best_fit = fit2.total_oc_delay([], fit2.data.m1, fit2.data.m2,
                                       fit2.data.inc, e_th, new_Mintimes)
        best_fit_left = best_fit * factor_left

    ax_left.plot(e_th, best_fit_left, color="r", label="Best Fit", linewidth=3)
    ax_left.axhline(y=0, color="black", linestyle="--", alpha=0.4, linewidth=1)
    
    # Set x-axis limits for the main plot (do not force extension)
    if x_lim is None:
        ax_left.set_xlim(e_th_min, e_th_max)
    else:
        ax_left.set_xlim(x_lim)

    # Set y-axis limits for the main plot
    if y_lim is not None:
        ax_left.set_ylim(y_lim)
    else:
        left_data_min = oc_left.min()
        left_data_max = oc_left.max()
        if left_data_max == left_data_min:
            left_data_min -= 1e-7
            left_data_max += 1e-7
        pad_left = (left_data_max - left_data_min) * 0.05
        ax_left.set_ylim(left_data_min - pad_left, left_data_max + pad_left)

    # Setup right y-axis if requested
    if y_axis_right:
        ax_right = ax_left.twinx()
        if y_lim_right is not None:
            ax_right.set_ylim(y_lim_right)
        else:
            left_min, left_max = ax_left.get_ylim()
            ratio = factor_right / factor_left
            ax_right.set_ylim(left_min * ratio, left_max * ratio)
    else:
        ax_right = None

    ax_left.yaxis.set_major_locator(MaxNLocator(nbins=7))
    left_range_val = ax_left.get_ylim()[1] - ax_left.get_ylim()[0]
    fmt_left = dynamic_format(left_range_val)
    ax_left.yaxis.set_major_formatter(FormatStrFormatter(fmt_left))
    ax_left.tick_params(axis='y', labelsize=tick_size_y_left)
    left_label = "O-C (ms)" if y_axis_left.lower() == "millisecond" else f"O-C ({y_axis_left.title()})"
    ax_left.set_ylabel(left_label, fontsize=label_size)
    
    if ax_right:
        ax_right.yaxis.set_major_locator(MaxNLocator(nbins=7))
        right_range_val = ax_right.get_ylim()[1] - ax_right.get_ylim()[0]
        fmt_right = dynamic_format(right_range_val)
        ax_right.yaxis.set_major_formatter(FormatStrFormatter(fmt_right))
        ax_right.tick_params(axis='y', labelsize=tick_size_y_right)
        right_label = "O-C (ms)" if y_axis_right.lower() == "millisecond" else f"O-C ({y_axis_right.title()})"
        ax_right.set_ylabel(right_label, fontsize=label_size)

    # Apply bottom x-axis formatting
    _apply_bottom_xaxis(ax_left, x_axis_bot)
    
    # ---- Top x-axis: Ticks on January 1 boundaries per your new rules ----
    if x_axis_top is not None:
        ax_top = ax_left.twiny()
        # Do not extend the main x-range; use the current limits
        ax_top.set_xlim(ax_left.get_xlim())
        ax_top.tick_params(axis='x', labelsize=tick_size_x_top)
        ax_top.xaxis.set_ticks_position('top')
        if x_axis_top.lower() == "year":
            ref_mintime = getattr(fit2.data, "Ref_mintime", 0)
            ref_period  = getattr(fit2.data, "Ref_period", 1)
            x_min, x_max = ax_left.get_xlim()
            bjd_min = ref_mintime + x_min * ref_period
            bjd_max = ref_mintime + x_max * ref_period
            yr_min = approximate_year_from_bjd(bjd_min)
            yr_max = approximate_year_from_bjd(bjd_max)
            # Per your instruction: first tick = floor(yr_min) + 1, last tick = ceil(yr_max)
            first_tick = int(np.floor(yr_min)) + 1
            last_tick  = int(np.ceil(yr_max))
            # Try to see if (last_tick - first_tick) is divisible by 8, then 7, then 6.
            divisor = None
            for d in [8, 7, 6]:
                if (last_tick - first_tick) % d == 0:
                    divisor = d
                    break
            # If not divisible, reduce last_tick by 1 until one works.
            while divisor is None and last_tick > first_tick:
                last_tick -= 1
                for d in [8, 7, 6]:
                    if (last_tick - first_tick) % d == 0:
                        divisor = d
                        break
            if divisor is not None:
                interval = (last_tick - first_tick) // divisor
                tick_years = [first_tick + i * interval for i in range(divisor + 1)]
            else:
                # Fallback: simply use first_tick and last_tick
                tick_years = [first_tick, last_tick]
            # Convert each tick year (January 1) to epoch units.
            top_positions = []
            for yr in tick_years:
                bjd_val = approximate_bjd_from_year(yr)
                epoch_val = (bjd_val - ref_mintime) / ref_period
                top_positions.append(epoch_val)
            ax_top.xaxis.set_major_locator(FixedLocator(top_positions))
            ax_top.xaxis.set_major_formatter(FixedFormatter([str(yr) for yr in tick_years]))
            ax_top.set_xlabel("Year", fontsize=label_size)
        elif x_axis_top.lower() == "bjd":
            ax_top.set_xlabel("BJD", fontsize=label_size)
        elif x_axis_top.lower() == "epoch":
            ax_top.set_xlabel("Cycle", fontsize=label_size)

    # Plot other models if provided
    if other_models and isinstance(other_models, list):
        fit2b = copy.deepcopy(fit2)
        for om in other_models:
            fit2b.model = om
            bf2 = fit2b.total_oc_delay([], fit2.data.m1, fit2.data.m2,
                                        fit2.data.inc, e_th, new_Mintimes)
            ax_left.plot(e_th, bf2 * factor_left, label="Other Model", linewidth=1)

    # Draw legend if requested
    if draw_legend:
        if isinstance(legend_position, tuple):
            if isinstance(legend_position[0], str):
                loc_used, bbox_used = legend_position
                ax_left.legend(loc=loc_used, bbox_to_anchor=bbox_used, fontsize=legend_size,
                               ncol=legend_shape[0], frameon=False, title="Data Group")
            else:
                ax_left.legend(loc="center", bbox_to_anchor=legend_position, fontsize=legend_size,
                               ncol=legend_shape[0], frameon=False, title="Data Group")
        else:
            if legend_position.lower() == "best":
                ax_left.legend(loc="best", fontsize=legend_size,
                               ncol=legend_shape[0], frameon=False, title="Data Group")
            else:
                loc_dict = {"top": ("upper center", (0.5, 1.02)),
                            "bottom": ("upper center", (0.5, -0.2)),
                            "left": ("center left", (-0.1, 0.5)),
                            "right": ("center right", (1.1, 0.5))}
                if legend_position.lower() in loc_dict:
                    loc, bbox = loc_dict[legend_position.lower()]
                    ax_left.legend(loc=loc, bbox_to_anchor=bbox, fontsize=legend_size,
                                   ncol=legend_shape[0], frameon=False, title="Data Group")
                else:
                    ax_left.legend(loc="best", fontsize=legend_size,
                                   ncol=legend_shape[0], frameon=False, title="Data Group")

    # Residuals subplot if requested
    if res_plot:
        df_reset = fit2.data.df.reset_index(drop=True)
        epochs = df_reset["Ecorr"].values
        observed = df_reset["OC"].values * factor_left
        predicted = np.interp(epochs, e_th, best_fit_left)
        residuals = observed - predicted
        errors = df_reset["Errors"].values * factor_left

        grouped_res = df_reset.groupby("Data_group")
        for name, grp in grouped_res:
            idx = grp.index
            x_vals = epochs[idx]
            y_vals = residuals[idx]
            y_errs = errors[idx]
            marker_here = group_shape_dict.get(name, ".")
            size_here = group_size_dict.get(name, 5)
            ax_res.errorbar(x_vals, y_vals, yerr=y_errs, fmt=marker_here,
                            linestyle='none', markersize=size_here, alpha=1,
                            elinewidth=1.2, capsize=2, color=group_color_dict.get(name))
        ax_res.axhline(0, color="black", linestyle="--", linewidth=1)
        _apply_bottom_xaxis(ax_res, x_axis_bot)
        
        res_min = residuals.min()
        res_max = residuals.max()
        if res_max == res_min:
            res_min -= 1e-7
            res_max += 1e-7
        pad_res = (res_max - res_min) * 0.05
        ax_res.set_ylim(res_min - pad_res, res_max + pad_res)

        left_label_res = "Residuals (ms)" if y_axis_left.lower() == "millisecond" else f"Residuals ({y_axis_left.title()})"
        ax_res.set_ylabel(left_label_res, fontsize=label_size)
        if x_lim is not None:
            ax_res.set_xlim(x_lim)
        else:
            ax_res.set_xlim(e_th_min, e_th_max)
        ax_res.yaxis.set_major_locator(MaxNLocator(nbins=7))
        res_range = ax_res.get_ylim()[1] - ax_res.get_ylim()[0]
        fmt_res = dynamic_format(res_range)
        ax_res.yaxis.set_major_formatter(FormatStrFormatter(fmt_res))
        ax_res.tick_params(axis='y', labelsize=tick_size_y_left)
        
        if y_axis_right:
            ax_res_right = ax_res.twinx()
            ratio = factor_right / factor_left
            left_ylim = ax_res.get_ylim()
            ax_res_right.set_ylim(left_ylim[0] * ratio, left_ylim[1] * ratio)
            ax_res_right.yaxis.set_major_locator(MaxNLocator(nbins=7))
            res_range_right = ax_res_right.get_ylim()[1] - ax_res_right.get_ylim()[0]
            fmt_right_res = dynamic_format(res_range_right)
            ax_res_right.yaxis.set_major_formatter(FormatStrFormatter(fmt_right_res))
            ax_res_right.tick_params(axis='y', labelsize=tick_size_y_right)
            right_label_res = "Residuals (ms)" if y_axis_right.lower() == "millisecond" else f"Residuals ({y_axis_right.title()})"
            ax_res_right.set_ylabel(right_label_res, fontsize=label_size)
            
        ax_res.set_title("Residuals", fontsize=label_size)

    if save_plots:
        plt.savefig(f"{fit2.data.object_name}_errorbar_plot_{outsuffix}.png",
                    dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
