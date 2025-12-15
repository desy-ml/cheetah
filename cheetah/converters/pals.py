import yaml
import numpy as np
from typing import Dict, Any, List, Union

class CheetahToPALSTranslator:
    """
    Translator to convert Cheetah lattice JSON files to PALS format.
    
    Based on the PALS specification: https://pals-project.readthedocs.io/
    """
    
    def __init__(self):
        # Map Cheetah element types to PALS element kinds
        self.element_mapping = {
            'Drift': 'Drift',
            'Quadrupole': 'Quadrupole',
            'Dipole': 'RBend',  # Assuming rectangular bends by default
            'BPM': 'Instrument',  # BPMs are diagnostic instruments
            'Marker': 'Marker',
            'Screen': 'Instrument',  # Screens are diagnostic instruments
            'Aperture': 'Mask',  # Apertures are typically masks/collimators
            'HorizontalCorrector': 'Kicker',  # Corrector magnets are kickers
            'VerticalCorrector': 'Kicker',
            'TransverseDeflectingCavity': 'RFCavity',  # RF cavity variant
        }
    
    def convert_segment_to_pals(self, segment, lattice_name="Converted_Lattice"):
        """
        Convert a Cheetah Segment to PALS format.
        
        Args:
            segment: Cheetah Segment object
            lattice_name: Name for the PALS lattice
            
        Returns:
            dict: PALS-formatted lattice dictionary
        """
        pals_elements = []
        
        # Add BeginningEle element (required as first element in PALS)
        pals_elements.append({
            'beginning': {
                'kind': 'BeginningEle'
            }
        })
        
        # Convert each element in the segment
        for i, element in enumerate(segment.elements):
            pals_element = self._convert_element(element, i)
            if pals_element:
                pals_elements.append(pals_element)
        
        # Add final marker (required as last element in PALS)
        pals_elements.append({
            'end_marker': {
                'kind': 'Marker'
            }
        })
        
        # Create the beamline
        beamline_name = f"{lattice_name}_line"
        beamline = {
            beamline_name: {
                'kind': 'BeamLine',
                'line': [list(elem.keys())[0] for elem in pals_elements]
            }
        }
        
        # Create the main lattice structure
        lattice = {
            'Lattice': {
                'name': lattice_name,
                'branches': [beamline_name]
            }
        }
        
        # Combine all elements
        pals_dict = {}
        for elem in pals_elements:
            pals_dict.update(elem)
        pals_dict.update(beamline)
        pals_dict.update(lattice)
        
        return pals_dict
    
    def _convert_element(self, element, index: int) -> Dict[str, Any]:
        """
        Convert a single Cheetah element to PALS format.
        
        Args:
            element: Cheetah element object
            index: Element index for naming
            
        Returns:
            dict: PALS element dictionary
        """
        element_type = type(element).__name__
        
        if element_type not in self.element_mapping:
            print(f"Warning: Unsupported element type {element_type}, skipping...")
            return None
        
        pals_kind = self.element_mapping[element_type]
        
        # Generate element name
        if hasattr(element, 'name') and element.name:
            element_name = element.name
        else:
            element_name = f"{element_type.lower()}_{index}"
        
        # Base element structure
        pals_element = {
            'kind': pals_kind
        }
        
        # Add length if element has it
        if hasattr(element, 'length') and element.length is not None:
            length_val = float(element.length.item()) if hasattr(element.length, 'item') else float(element.length)
            pals_element['length'] = length_val
        
        # Add element-specific parameters
        if element_type == 'Quadrupole':
            pals_element.update(self._convert_quadrupole(element))
        elif element_type == 'Dipole':
            pals_element.update(self._convert_dipole(element))
        elif element_type in ['HorizontalCorrector', 'VerticalCorrector']:
            pals_element.update(self._convert_corrector(element))
        elif element_type == 'TransverseDeflectingCavity':
            pals_element.update(self._convert_cavity(element))
        elif element_type == 'BPM':
            pals_element.update(self._convert_bpm(element))
        elif element_type == 'Screen':
            pals_element.update(self._convert_screen(element))
        elif element_type == 'Aperture':
            pals_element.update(self._convert_aperture(element))
        
        # Add metadata
        meta_p = {
            'description': f"Converted from Cheetah {element_type}"
        }
        if hasattr(element, 'name') and element.name:
            meta_p['alias'] = element.name
        
        pals_element['MetaP'] = meta_p
        
        return {element_name: pals_element}
    
    def _convert_quadrupole(self, element) -> Dict[str, Any]:
        """Convert Cheetah Quadrupole to PALS format."""
        params = {}
        
        # Magnetic multipole parameters
        if hasattr(element, 'k1') and element.k1 is not None:
            k1_val = float(element.k1.item()) if hasattr(element.k1, 'item') else float(element.k1)
            length_val = float(element.length.item()) if hasattr(element.length, 'item') else float(element.length)
            
            params['MagneticMultipoleP'] = {
                'Kn1L': k1_val * length_val  # PALS uses integrated strength
            }
        
        return params
    
    def _convert_dipole(self, element) -> Dict[str, Any]:
        """Convert Cheetah Dipole to PALS format."""
        params = {}
        
        # Bend parameters
        if hasattr(element, 'angle') and element.angle is not None:
            angle_val = float(element.angle.item()) if hasattr(element.angle, 'item') else float(element.angle)
            params['BendP'] = {
                'angle_ref': angle_val
            }
        
        # Add magnetic multipole if there are higher order terms
        if hasattr(element, 'k1') and element.k1 is not None:
            k1_val = float(element.k1.item()) if hasattr(element.k1, 'item') else float(element.k1)
            length_val = float(element.length.item()) if hasattr(element.length, 'item') else float(element.length)
            if k1_val != 0:
                params['MagneticMultipoleP'] = {
                    'Kn1L': k1_val * length_val
                }
        
        return params
    
    def _convert_corrector(self, element) -> Dict[str, Any]:
        """Convert Cheetah Corrector to PALS format."""
        params = {}
        
        # Determine if horizontal or vertical corrector
        element_type = type(element).__name__
        
        if hasattr(element, 'kick') and element.kick is not None:
            kick_val = float(element.kick.item()) if hasattr(element.kick, 'item') else float(element.kick)
            
            # Kicker parameters (simplified)
            if element_type == 'HorizontalCorrector':
                params['KickerP'] = {
                    'hkick': kick_val
                }
            else:  # VerticalCorrector
                params['KickerP'] = {
                    'vkick': kick_val
                }
        
        return params
    
    def _convert_cavity(self, element) -> Dict[str, Any]:
        """Convert Cheetah TransverseDeflectingCavity to PALS format."""
        params = {}
        
        # RF parameters
        rf_params = {}
        
        if hasattr(element, 'voltage') and element.voltage is not None:
            voltage_val = float(element.voltage.item()) if hasattr(element.voltage, 'item') else float(element.voltage)
            rf_params['voltage'] = voltage_val
        
        if hasattr(element, 'phase') and element.phase is not None:
            phase_val = float(element.phase.item()) if hasattr(element.phase, 'item') else float(element.phase)
            rf_params['phi'] = phase_val
        
        if hasattr(element, 'frequency') and element.frequency is not None:
            freq_val = float(element.frequency.item()) if hasattr(element.frequency, 'item') else float(element.frequency)
            rf_params['frequency'] = freq_val
        
        if rf_params:
            params['RFP'] = rf_params
        
        return params
    
    def _convert_bpm(self, element) -> Dict[str, Any]:
        """Convert Cheetah BPM to PALS format."""
        # BPMs typically don't have parameters beyond position
        return {}
    
    def _convert_screen(self, element) -> Dict[str, Any]:
        """Convert Cheetah Screen to PALS format."""
        # Screens typically don't have specific parameters
        return {}
    
    def _convert_aperture(self, element) -> Dict[str, Any]:
        """Convert Cheetah Aperture to PALS format."""
        params = {}
        
        # Aperture parameters
        aperture_params = {}
        
        if hasattr(element, 'x_max') and element.x_max is not None:
            x_max_val = float(element.x_max.item()) if hasattr(element.x_max, 'item') else float(element.x_max)
            aperture_params['x_limit'] = [-x_max_val, x_max_val]
        
        if hasattr(element, 'y_max') and element.y_max is not None:
            y_max_val = float(element.y_max.item()) if hasattr(element.y_max, 'item') else float(element.y_max)
            aperture_params['y_limit'] = [-y_max_val, y_max_val]
        
        if aperture_params:
            params['ApertureP'] = aperture_params
        
        return params
    
    def save_to_yaml(self, pals_dict: Dict[str, Any], filename: str):
        """
        Save PALS dictionary to YAML file.
        
        Args:
            pals_dict: PALS lattice dictionary
            filename: Output filename (should end with .pals.yaml)
        """
        with open(filename, 'w') as f:
            yaml.dump(pals_dict, f, default_flow_style=False, sort_keys=False)
    
    def save_to_json(self, pals_dict: Dict[str, Any], filename: str):
        """
        Save PALS dictionary to JSON file.
        
        Args:
            pals_dict: PALS lattice dictionary  
            filename: Output filename (should end with .pals.json)
        """
        import json
        with open(filename, 'w') as f:
            json.dump(pals_dict, f, indent=2)

# Create translator instance
translator = CheetahToPALSTranslator()