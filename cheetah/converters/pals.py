"""
Cheetah to PALS converter module.

This module provides functions to convert Cheetah lattice elements and segments
to the PALS (Particle Accelerator Lattice Standard) format.

Based on the PALS specification: https://pals-project.readthedocs.io/
"""
import json
import warnings
from pathlib import Path
from typing import Any, Dict, Union

import torch
import yaml

import cheetah
from cheetah.utils import UnknownElementWarning


def convert_element_to_pals(
    element: "cheetah.Element",
    name: str = None,
) -> Dict[str, Any]:
    """
    Convert a Cheetah element to PALS format.
    
    :param element: Cheetah element to convert.
    :param name: Name for the element. If None, uses element.name or generates one.
    :return: Dictionary representing the element in PALS format.
    """
    element_type = type(element).__name__
    
    # Map Cheetah element types to PALS element kinds
    element_mapping = {
        'Drift': 'Drift',
        'Quadrupole': 'Quadrupole',
        'Dipole': 'RBend',
        'RBend': 'RBend',
        'BPM': 'Instrument',
        'Marker': 'Marker',
        'Screen': 'Instrument',
        'Aperture': 'Mask',
        'HorizontalCorrector': 'Kicker',
        'VerticalCorrector': 'Kicker',
        'TransverseDeflectingCavity': 'RFCavity',
        'Cavity': 'RFCavity',
        'Solenoid': 'Solenoid',
        'Sextupole': 'Sextupole',
    }
    
    if element_type not in element_mapping:
        warnings.warn(
            f"Element type '{element_type}' is not supported in PALS conversion. "
            "Converting to Marker element.",
            UnknownElementWarning,
        )
        element_mapping[element_type] = 'Marker'
    
    pals_kind = element_mapping[element_type]
    
    # Generate element name
    element_name = name or getattr(element, 'name', None) or f"{element_type.lower()}_0"
    
    # Base element structure
    pals_element = {'kind': pals_kind}
    
    # Add length if element has it
    if hasattr(element, 'length') and element.length is not None:
        length_val = _extract_tensor_value(element.length)
        pals_element['length'] = length_val
    
    # Add element-specific parameters based on type
    if element_type == 'Quadrupole':
        pals_element.update(_convert_quadrupole_params(element))
    elif element_type in ['Dipole', 'RBend']:
        pals_element.update(_convert_dipole_params(element))
    elif element_type in ['HorizontalCorrector', 'VerticalCorrector']:
        pals_element.update(_convert_corrector_params(element))
    elif element_type in ['TransverseDeflectingCavity', 'Cavity']:
        pals_element.update(_convert_cavity_params(element))
    elif element_type == 'Aperture':
        pals_element.update(_convert_aperture_params(element))
    elif element_type == 'Solenoid':
        pals_element.update(_convert_solenoid_params(element))
    elif element_type == 'Sextupole':
        pals_element.update(_convert_sextupole_params(element))
    
    return {element_name: pals_element}


def convert_segment_to_pals(
    segment: "cheetah.Segment",
    lattice_name: str = "Converted_Lattice",
) -> Dict[str, Any]:
    """
    Convert a Cheetah Segment to PALS format.
    
    :param segment: Cheetah Segment object to convert.
    :param lattice_name: Name for the PALS lattice.
    :return: Dictionary representing the lattice in PALS format.
    """
    pals_elements = {}
    element_names = []
    
    # Add BeginningEle element (required as first element in PALS)
    beginning_name = 'beginning'
    pals_elements[beginning_name] = {'kind': 'BeginningEle'}
    element_names.append(beginning_name)
    
    # Convert each element in the segment
    for i, element in enumerate(segment.elements):
        # Generate unique name if element doesn't have one
        base_name = getattr(element, 'name', None) or f"{type(element).__name__.lower()}_{i}"
        
        # Ensure unique names
        element_name = base_name
        counter = 1
        while element_name in pals_elements:
            element_name = f"{base_name}_{counter}"
            counter += 1
        
        pals_element_dict = convert_element_to_pals(element, element_name)
        pals_elements.update(pals_element_dict)
        element_names.append(element_name)
    
    # Add final marker (required as last element in PALS)
    end_marker_name = 'end_marker'
    pals_elements[end_marker_name] = {'kind': 'Marker'}
    element_names.append(end_marker_name)
    
    # Create the beamline
    beamline_name = f"{lattice_name}_line"
    pals_elements[beamline_name] = {
        'kind': 'BeamLine',
        'line': element_names
    }
    
    # Create the main lattice structure
    pals_elements['Lattice'] = {
        'name': lattice_name,
        'branches': [beamline_name]
    }
    
    return pals_elements


def _extract_tensor_value(tensor_value: Union[torch.Tensor, float, int]) -> float:
    """Extract float value from tensor or numeric type."""
    if hasattr(tensor_value, 'item'):
        return float(tensor_value.item())
    else:
        return float(tensor_value)


def _convert_quadrupole_params(element) -> Dict[str, Any]:
    """Convert Cheetah Quadrupole parameters to PALS format."""
    params = {}
    
    if hasattr(element, 'k1') and element.k1 is not None:
        k1_val = _extract_tensor_value(element.k1)
        length_val = _extract_tensor_value(element.length)
        
        params['MagneticMultipoleP'] = {
            'Kn1L': k1_val * length_val  # PALS uses integrated strength
        }
    
    return params


def _convert_dipole_params(element) -> Dict[str, Any]:
    """Convert Cheetah Dipole parameters to PALS format."""
    params = {}
    
    # Bend parameters
    if hasattr(element, 'angle') and element.angle is not None:
        angle_val = _extract_tensor_value(element.angle)
        params['BendP'] = {'angle_ref': angle_val}
    
    # Add magnetic multipole if there are higher order terms
    if hasattr(element, 'k1') and element.k1 is not None:
        k1_val = _extract_tensor_value(element.k1)
        if k1_val != 0:
            length_val = _extract_tensor_value(element.length)
            if 'MagneticMultipoleP' not in params:
                params['MagneticMultipoleP'] = {}
            params['MagneticMultipoleP']['Kn1L'] = k1_val * length_val
    
    return params


def _convert_corrector_params(element) -> Dict[str, Any]:
    """Convert Cheetah Corrector parameters to PALS format."""
    params = {}
    element_type = type(element).__name__
    
    if hasattr(element, 'angle') and element.angle is not None:
        angle_val = _extract_tensor_value(element.angle)
        
        if element_type == 'HorizontalCorrector':
            params['KickerP'] = {'hkick': angle_val}
        else:  # VerticalCorrector
            params['KickerP'] = {'vkick': angle_val}
    
    return params


def _convert_cavity_params(element) -> Dict[str, Any]:
    """Convert Cheetah Cavity parameters to PALS format."""
    params = {}
    rf_params = {}
    
    if hasattr(element, 'voltage') and element.voltage is not None:
        rf_params['voltage'] = _extract_tensor_value(element.voltage)
    
    if hasattr(element, 'phase') and element.phase is not None:
        rf_params['phi'] = _extract_tensor_value(element.phase)
    
    if hasattr(element, 'frequency') and element.frequency is not None:
        rf_params['frequency'] = _extract_tensor_value(element.frequency)
    
    if rf_params:
        params['RFP'] = rf_params
    
    return params


def _convert_aperture_params(element) -> Dict[str, Any]:
    """Convert Cheetah Aperture parameters to PALS format."""
    params = {}
    aperture_params = {}
    
    if hasattr(element, 'x_max') and element.x_max is not None:
        x_max_val = _extract_tensor_value(element.x_max)
        aperture_params['x_limit'] = [-x_max_val, x_max_val]
    
    if hasattr(element, 'y_max') and element.y_max is not None:
        y_max_val = _extract_tensor_value(element.y_max)
        aperture_params['y_limit'] = [-y_max_val, y_max_val]
    
    if aperture_params:
        params['ApertureP'] = aperture_params
    
    return params


def _convert_solenoid_params(element) -> Dict[str, Any]:
    """Convert Cheetah Solenoid parameters to PALS format."""
    params = {}
    
    if hasattr(element, 'k') and element.k is not None:
        k_val = _extract_tensor_value(element.k)
        length_val = _extract_tensor_value(element.length)
        
        params['SolenoidP'] = {
            'KsL': k_val * length_val  # Integrated solenoid strength
        }
    
    return params


def _convert_sextupole_params(element) -> Dict[str, Any]:
    """Convert Cheetah Sextupole parameters to PALS format."""
    params = {}
    
    if hasattr(element, 'k2') and element.k2 is not None:
        k2_val = _extract_tensor_value(element.k2)
        length_val = _extract_tensor_value(element.length)
        
        params['MagneticMultipoleP'] = {
            'Kn2L': k2_val * length_val  # Integrated sextupole strength
        }
    
    return params


def save_pals_to_yaml(
    pals_dict: Dict[str, Any],
    filename: Union[str, Path],
) -> None:
    """
    Save PALS dictionary to YAML file.
    
    :param pals_dict: PALS lattice dictionary.
    :param filename: Output filename (should end with .pals.yaml).
    """
    with open(filename, 'w') as f:
        yaml.dump(pals_dict, f, default_flow_style=False, sort_keys=False)


def save_pals_to_json(
    pals_dict: Dict[str, Any],
    filename: Union[str, Path],
) -> None:
    """
    Save PALS dictionary to JSON file.
    
    :param pals_dict: PALS lattice dictionary.
    :param filename: Output filename (should end with .pals.json).
    """
    with open(filename, 'w') as f:
        json.dump(pals_dict, f, indent=2)