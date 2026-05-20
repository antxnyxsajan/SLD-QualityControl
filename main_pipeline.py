import sys
from importlib import import_module

# Dynamically import modules that start with numbers
structural_module = import_module('step1_structuralValidator')
acoustic_module = import_module('step2_acousticValidator')

def run_pipeline(rttm_filepath, audio_filepath=None):
    print(f"--- Starting Quality Control Pipeline ---")
    print(f"RTTM File: {rttm_filepath}")
    
    # 1. Structural Validation
    print("\n[1] Running Structural Validation...")
    # Initialize RTTM validator
    rttm_validator = structural_module.RTTMValidator()
    struct_results = rttm_validator.validate_file(rttm_filepath)
    
    print(f"Structural Validation Passed: {struct_results['is_valid']}")
    if struct_results['errors']:
        print("Structural Errors:")
        for err in struct_results['errors']:
            print(f"  - {err}")
    print(f"\n - SHORT SEGMENT COUNT :{struct_results['short_segment_count']}")
    if struct_results['warnings']:
        print("\nStructural Warnings:")
        for warn in struct_results['warnings']:
            print(f"  - {warn}")
    print(f"\n - OVERLAPPING COUNT :{struct_results['overlaped segments']}")
            
    # 2. Acoustic Validation
    print("\n[2] Running Acoustic Validation...")
    # Initialize Acoustic validator
    acoustic_validator = acoustic_module.AcousticValidator()
    
    # Assuming acoustic validator might need an audio file in the future
    # acoustic_results = acoustic_validator.validate(audio_filepath)
    acoustic_results = acoustic_validator.validate(rttm_filepath)
    
    print(f"Acoustic Validation Passed: {acoustic_results['is_valid']}")
    if acoustic_results['errors']:
        print("Acoustic Errors:")
        for err in acoustic_results['errors']:
            print(f"  - {err}")
            
    print("\n--- Pipeline Finished ---")

if __name__ == "__main__":
    rttm_test_file = "test.rttm"
    if len(sys.argv) > 1:
        rttm_test_file = sys.argv[1]
    
    run_pipeline(rttm_test_file)
