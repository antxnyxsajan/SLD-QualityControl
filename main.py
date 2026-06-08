import argparse
from importlib import import_module

structural_module = import_module('step1_structuralValidator')
acoustic_module = import_module('step2_acousticValidator')
language_module = import_module('step3_languageValidator')

# Passing threshold for the dataset (95% agreement required)
PASSING_THRESHOLD = 95.0

def run_pipeline(rttm_filepath, audio_filepath=None, rttml_filepath=None, silent=False):
    if not silent:
        print(f"--- Starting Ground Truth QA Auditor ---")
    
    rttm_validator = structural_module.RTTMValidator()

    # =============================================
    # STEP 1.1: Structural Validation (Speaker RTTM)
    # =============================================
    if not silent:
        print("\n[1.1] Running Structural Validation (Speaker RTTM)...")
        print(f"File: {rttm_filepath}")
    struct_results = rttm_validator.validate_file(rttm_filepath)
    
    if not silent:
        print(f"Structural Integrity: {'PASSED' if struct_results['is_valid'] else 'ISSUES FOUND'}")
        print(f"Structural Score: {struct_results['score']}/100")
        if struct_results['errors']:
            for err in struct_results['errors']:
                print(f"  Line {err['line']} [{err['severity']}]: {err['message']}")
        if struct_results['warnings']:
            for warn in struct_results['warnings']:
                print(f"  Line {warn['line']} [{warn['severity']}]: {warn['message']}")

    # =============================================
    # STEP 1.2: Structural Validation (Language RTTM)
    # =============================================
    structr_results = None
    if rttml_filepath:
        if not silent:
            print("\n[1.2] Running Structural Validation (Language RTTM)...")
            print(f"File: {rttml_filepath}")
        rttm_validator_lang = structural_module.RTTMValidator()
        structr_results = rttm_validator_lang.validate_file(rttml_filepath)
        if not silent:
            print(f"Structural Integrity: {'PASSED' if structr_results['is_valid'] else 'ISSUES FOUND'}")
            print(f"Structural Score: {structr_results['score']}/100")
            if structr_results['errors']:
                for err in structr_results['errors']:
                    print(f"  Line {err['line']} [{err['severity']}]: {err['message']}")
            if structr_results['warnings']:
                for warn in structr_results['warnings']:
                    print(f"  Line {warn['line']} [{warn['severity']}]: {warn['message']}")

    # =============================================
    # STEP 2: Speaker Diarization Validation
    # =============================================
    if not silent:
        print("\n[2] Running Speaker Diarization Validation...")
    acoustic_validator = acoustic_module.AcousticValidator()
    acoustic_results = acoustic_validator.validate(rttm_filepath, audio_filepath, struct_results)
    
    spk_agreement = acoustic_results['agreement_rate']
    if not silent:
        print(f"Speaker Annotation Credibility: {spk_agreement:.2%} Agreement ({acoustic_results['total_comparisons']} comparisons)")
        print(f"Average Cosine Similarity: {acoustic_results['avg_confidence']:.4f}")
        print(f"F1 Score: {acoustic_results['f1_score']:.4f}")
        print(f"Anomalies: {acoustic_results['severe_count']} SEVERE | {acoustic_results['moderate_count']} MODERATE")
        if acoustic_results.get('warnings'):
            for warn in acoustic_results['warnings']:
                print(f"  [WARNING] {warn}")

    # =============================================
    # STEP 3: Language Diarization Validation
    # =============================================
    lang_results = None
    if rttml_filepath:
        if not silent:
            print("\n[3] Running Language Diarization Validation...")
        language_validator = language_module.LanguageValidator()
        lang_results = language_validator.validate(rttml_filepath, audio_filepath, structr_results)
        
        lang_agreement = lang_results['agreement_rate']
        if not silent:
            print(f"Language Annotation Credibility: {lang_agreement:.2%} Agreement ({lang_results['total_comparisons']} comparisons)")
            print(f"F1 Score: {lang_results['f1_score']:.4f}")
            print(f"Anomalies: {lang_results['severe_count']} SEVERE | {lang_results['moderate_count']} MODERATE | {lang_results.get('ambiguous_count', 0)} AMBIGUOUS (possible code-switch)")
            if lang_results.get('warnings'):
                for warn in lang_results['warnings']:
                    print(f"  [WARNING] {warn}")

    # =============================================
    # FINAL QA AUDIT REPORT
    # =============================================
    if not silent:
        print("\n" + "=" * 60)
        print("GROUND TRUTH QA AUDIT REPORT")
        print("=" * 60)
        
        # --- Speaker RTTM Report ---
        print("\nSPEAKER RTTM QUALITY")
        print(f"  Structural Score:         {struct_results['score']}/100")
        print(f"  Annotation Agreement:     {spk_agreement:.2%}")
        print(f"  F1 Score:                 {acoustic_results['f1_score']:.4f}")
        print(f"  Severe Anomalies:         {acoustic_results['severe_count']}")
        print(f"  Moderate Anomalies:       {acoustic_results['moderate_count']}")
        
        # Chronological sorted SEVERE anomalies only
        spk_severe = sorted(
            [a for a in acoustic_results['anomalies'] if a['severity'] == 'SEVERE'],
            key=lambda x: x['line']
        )
        if spk_severe:
            print("\n  SEVERE ISSUES (Requires Manual Review):")
            for a in spk_severe:
                print(f"    Line {a['line']}: {a['message']}")
                
        # Chronological sorted MODERATE anomalies
        spk_moderate = sorted(
            [a for a in acoustic_results['anomalies'] if a['severity'] == 'MODERATE'],
            key=lambda x: x['line']
        )
        if spk_moderate:
            print("\n  MODERATE ISSUES (Warnings / Edge Cases):")
            for a in spk_moderate:
                print(f"    Line {a['line']}: {a['message']}")
        
        spk_pass = spk_agreement * 100 >= PASSING_THRESHOLD
        print(f"\n  STATUS: {'PASSED ✓' if spk_pass else 'FAILED ✗ — Manual review required before use in training.'}")
        
        # --- Language RTTM Report ---
        if rttml_filepath and lang_results:
            print("\n" + "-" * 60)
            print("\nLANGUAGE RTTM QUALITY")
            print(f"  Structural Score:         {structr_results['score']}/100")
            print(f"  Annotation Agreement:     {lang_agreement:.2%}")
            print(f"  F1 Score:                 {lang_results['f1_score']:.4f}")
            print(f"  Severe Anomalies:         {lang_results['severe_count']}")
            print(f"  Moderate Anomalies:       {lang_results['moderate_count']}")
            print(f"  Ambiguous (Code-Switch):  {lang_results.get('ambiguous_count', 0)}")
            
            # Chronological sorted SEVERE anomalies only
            lang_severe = sorted(
                [a for a in lang_results['anomalies'] if a['severity'] == 'SEVERE'],
                key=lambda x: x['line']
            )
            if lang_severe:
                print("\n  SEVERE ISSUES (Requires Manual Review):")
                for a in lang_severe:
                    print(f"    Line {a['line']}: {a['message']}")
                    
            # Chronological sorted MODERATE anomalies
            lang_moderate = sorted(
                [a for a in lang_results['anomalies'] if a['severity'] == 'MODERATE'],
                key=lambda x: x['line']
            )
            if lang_moderate:
                print("\n  MODERATE ISSUES (Warnings / Edge Cases):")
                for a in lang_moderate:
                    print(f"    Line {a['line']}: {a['message']}")

            # Chronological sorted AMBIGUOUS segments (possible code-switches)
            lang_ambiguous = sorted(
                [a for a in lang_results['anomalies'] if a['severity'] == 'AMBIGUOUS'],
                key=lambda x: x['line']
            )
            if lang_ambiguous:
                print("\n  AMBIGUOUS (Possible Code-Switch — Manual Review Recommended):")
                for a in lang_ambiguous:
                    print(f"    Line {a['line']}: {a['message']}")
            
            lang_pass = lang_agreement * 100 >= PASSING_THRESHOLD
            print(f"\n  STATUS: {'PASSED ✓' if lang_pass else 'FAILED ✗ — Manual review required before use in training.'}")
        
        print("\n" + "=" * 60)

    # =============================================
    # Return structured results for batch_runner
    # =============================================
    spk_pass = spk_agreement * 100 >= PASSING_THRESHOLD
    lang_pass = True
    if rttml_filepath and lang_results:
        lang_agreement = lang_results['agreement_rate']
        lang_pass = lang_agreement * 100 >= PASSING_THRESHOLD

    return {
        # Speaker
        "spk_structural_score": struct_results['score'],
        "spk_f1":               acoustic_results['f1_score'],
        "spk_agreement":        acoustic_results['agreement_rate'],
        "spk_severe":           acoustic_results['severe_count'],
        "spk_moderate":         acoustic_results['moderate_count'],
        # Language (None if no language RTTM provided)
        "lang_structural_score": structr_results['score'] if structr_results else None,
        "lang_f1":               lang_results['f1_score'] if lang_results else None,
        "lang_agreement":        lang_results['agreement_rate'] if lang_results else None,
        "lang_severe":           lang_results['severe_count'] if lang_results else None,
        "lang_moderate":         lang_results['moderate_count'] if lang_results else None,
        "lang_ambiguous":        lang_results.get('ambiguous_count', 0) if lang_results else None,
        "label_map":             lang_results.get('label_map', {}) if lang_results else {},
        # Overall
        "passed":                spk_pass and lang_pass,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ground Truth QA Auditor for RTTM and Audio files.")
    parser.add_argument("--rttm", type=str, required=True, help="Path to the Speaker RTTM file.")
    parser.add_argument("--rttml", type=str, default=None, help="Path to the Language RTTM file (optional).")
    parser.add_argument("--audio", type=str, default=None, help="Path to the Audio WAV file (optional).")
    
    args = parser.parse_args()
    run_pipeline(args.rttm, args.audio, args.rttml)
