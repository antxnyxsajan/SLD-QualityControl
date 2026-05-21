import argparse
from importlib import import_module

# Dynamically import modules that start with numbers
structural_module = import_module('step1_structuralValidator')
acoustic_module = import_module('step2_acousticValidator')
language_module = import_module('step3_languageValidator')

def run_pipeline(rttm_filepath, audio_filepath=None, rttml_filepath=None):
    print(f"--- Starting Quality Control Pipeline ---")
    
    # Initialize RTTM validator
    rttm_validator = structural_module.RTTMValidator()

    # 1. Structural Validation (Speaker RTTM)
    print("\n[1] Running Structural Validation (Speaker RTTM)...")
    print(f"File: {rttm_filepath}")
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

    # Structural Validation (Language RTTM) - Optional
    structr_results = None
    if rttml_filepath:
        print("\n[1.5] Running Structural Validation (Language RTTM)...")
        print(f"File: {rttml_filepath}")
        structr_results = rttm_validator.validate_file(rttml_filepath)
        
        print(f"Structural Validation Passed: {structr_results['is_valid']}")
        if structr_results['errors']:
            print("Structural Errors:")
            for err in structr_results['errors']:
                print(f"  - {err}")
        print(f"\n - SHORT SEGMENT COUNT :{structr_results['short_segment_count']}")
        
        if structr_results['warnings']:
            print("\nStructural Warnings:")
            for warn in structr_results['warnings']:
                print(f"  - {warn}")
        print(f"\n - OVERLAPPING COUNT :{structr_results['overlaped segments']}")

    # 2. Acoustic Validation
    print("\n[2] Running Acoustic Validation...")
    acoustic_validator = acoustic_module.AcousticValidator()
    acoustic_results = acoustic_validator.validate(rttm_filepath, audio_filepath, struct_results)
    
    print(f"Acoustic Validation Passed: {acoustic_results['is_valid']}")
    if acoustic_results['errors']:
        print("Acoustic Errors:")
        for err in acoustic_results['errors']:
            print(f"  - {err}")
    if acoustic_results['warnings']:
        print("Acoustic Warnings:")
        for warn in acoustic_results['warnings']:
            print(f"  - {warn}")

    # 3. Language Validation - Only if Language RTTM is provided
    if rttml_filepath:
        print("\n[3] Running Language Validation...")
        language_validator = language_module.LanguageValidator()
        lang_results = language_validator.validate(rttml_filepath, audio_filepath, structr_results)
        
        print(f"Language Validation Passed: {lang_results['is_valid']}")
        if lang_results['errors']:
            print("Language Errors:")
            for err in lang_results['errors']:
                print(f"  - {err}")
        if lang_results['warnings']:
            print("Language Warnings:")
            for warn in lang_results['warnings']:
                print(f"  - {warn}")
    else:
        print("\n[3] Skipping Language Validation (No Language RTTM provided).")
            
    print("\n--- Pipeline Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quality Control Pipeline for RTTM and Audio files.")
    parser.add_argument("--rttm", type=str, default="test.rttm", help="Path to the Speaker RTTM file.")
    parser.add_argument("--rttml", type=str, default=None, help="Path to the Language RTTM file (optional).")
    parser.add_argument("--audio", type=str, default=None, help="Path to the Audio WAV file (optional).")
    
    args = parser.parse_args()
    
    run_pipeline(args.rttm, args.audio, args.rttml)
