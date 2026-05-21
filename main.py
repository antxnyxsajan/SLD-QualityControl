import argparse
from importlib import import_module

structural_module = import_module('step1_structuralValidator')
acoustic_module = import_module('step2_acousticValidator')
language_module = import_module('step3_languageValidator')

def run_pipeline(rttm_filepath, audio_filepath=None, rttml_filepath=None):
    print(f"--- Starting Quality Control Pipeline ---")
    
    rttm_validator = structural_module.RTTMValidator()

    # --- 1. Structural Validation (Speaker RTTM) ---
    print("\n[1.1] Running Structural Validation (Speaker RTTM)...")
    print(f"File: {rttm_filepath}")
    struct_results = rttm_validator.validate_file(rttm_filepath)
    
    print(f"Structural Validation Passed: {struct_results['is_valid']}")
    print(f"Structural Score: {struct_results['score']}/100")
    if struct_results['errors']:
        for err in struct_results['errors']:
            print(f"  - {err}")
    if struct_results['warnings']:
        for warn in struct_results['warnings']:
            print(f"  - {warn}")

    # --- Structural Validation (Language RTTM) ---
    structr_results = None
    if rttml_filepath:
        print("\n[1.2] Running Structural Validation (Language RTTM)...")
        structr_results = rttm_validator.validate_file(rttml_filepath)
        print(f"Structural Validation Passed: {structr_results['is_valid']}")
        print(f"Structural Score: {structr_results['score']}/100")
        if structr_results['errors']:
            for err in structr_results['errors']:
                print(f"  - {err}")

    # --- 2. Acoustic Validation ---
    print("\n[2] Running Speaker Diarization Validation...")
    acoustic_validator = acoustic_module.AcousticValidator()
    acoustic_results = acoustic_validator.validate(rttm_filepath, audio_filepath, struct_results)
    
    print(f"Speaker Validation Passed: {acoustic_results['is_valid']}")
    print(f"Speaker Accuracy Score: {acoustic_results['score']}/100 ({acoustic_results['accuracy']:.2%} over {acoustic_results.get('total_comparisons', 0)} comparisons)")
    print(f"Average Confidence (Cosine Similarity): {acoustic_results.get('avg_confidence', 0.0):.2f}")
    if acoustic_results['errors']:
        for err in acoustic_results['errors']:
            print(f"  - {err}")

    # --- 3. Language Validation ---
    lang_results = None
    if rttml_filepath:
        print("\n[3] Running Language Diarization Validation...")
        language_validator = language_module.LanguageValidator()
        lang_results = language_validator.validate(rttml_filepath, audio_filepath, structr_results)
        
        print(f"Language Validation Passed: {lang_results['is_valid']}")
        print(f"Language Accuracy Score: {lang_results['score']}/100 ({lang_results['accuracy']:.2%} over {lang_results.get('total_comparisons', 0)} comparisons)")
        if lang_results['errors']:
            for err in lang_results['errors']:
                print(f"  - {err}")
    
    # --- Final Scoring ---
    print("\n==================================================")
    print("FINAL QUALITY EVALUATION REPORT")
    print("==================================================")
    
    print("SPEAKER RTTM QUALITY")
    print(f" - Structural Score: {struct_results['score']}/100")
    print(f" - Validation Accuracy: {acoustic_results['score']}/100")
    speaker_overall = (struct_results['score'] + acoustic_results['score']) / 2
    print(f" -> OVERALL SPEAKER QUALITY: {speaker_overall:.1f}/100")
    
    if rttml_filepath and lang_results:
        print("\n--------------------------------------------------")
        print("LANGUAGE RTTM QUALITY")
        print(f" - Structural Score: {structr_results['score']}/100")
        print(f" - Validation Accuracy: {lang_results['score']}/100")
        lang_overall = (structr_results['score'] + lang_results['score']) / 2
        print(f" -> OVERALL LANGUAGE QUALITY: {lang_overall:.1f}/100")
        
    print("==================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quality Control Pipeline for RTTM and Audio files.")
    parser.add_argument("--rttm", type=str, default="test.rttm", help="Path to the Speaker RTTM file.")
    parser.add_argument("--rttml", type=str, default=None, help="Path to the Language RTTM file (optional).")
    parser.add_argument("--audio", type=str, default=None, help="Path to the Audio WAV file (optional).")
    
    args = parser.parse_args()
    run_pipeline(args.rttm, args.audio, args.rttml)
