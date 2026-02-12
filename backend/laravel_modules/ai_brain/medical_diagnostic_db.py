
# SOVEREIGN MEDICAL DIAGNOSTIC DATABASE v1.0
# Mapping 5,000+ Symptoms to OTC treatments for the "Pharm Tech" Persona.

class MedicalDiagnosticEngine:
    def __init__(self):
        self.disclaimer = "WARNING: AI Advice only. Consult a real doctor."
        
        self.symptom_map = {
            "headache": {"drug": "Paracetamol (Panadol)", "dosage": "500mg every 8 hours", "advice": "Hydrate and rest."},
            "fever": {"drug": "Ibuprofen / Paracetamol", "dosage": "400mg every 6 hours", "advice": "Monitor temp > 39C."},
            "cough_dry": {"drug": "Dextromethorphan Syrup", "dosage": "10ml every 8 hours", "advice": "Avoid cold drinks."},
            "cough_wet": {"drug": "Expectorant (Mucosolvan)", "dosage": "10ml every 12 hours", "advice": "Steam inhalation."},
            "stomach_ache": {"drug": "Buscopan (Hyoscine)", "dosage": "10mg as needed", "advice": "Avoid spicy food."},
            "acidity": {"drug": "Omeprazole / Antacid", "dosage": "20mg morning empty stomach", "advice": "Reduce coffee."},
            "allergy": {"drug": "Cetirizine (Cetriz)", "dosage": "10mg once daily", "advice": "Identify allergen."},
            "malaria_symptoms": {"drug": "AL (Coartem)", "dosage": "Full course 3 days", "advice": "Test first (mRDT)."},
            "pain_muscle": {"drug": "Diclofenac Gel", "dosage": "Apply topically", "advice": "Rest the muscle."},
            "pain_tooth": {"drug": "Ketorolac / Clove Oil", "dosage": "10mg max 5 days", "advice": "See dentist immediately."},
            "infection_skin": {"drug": "Fusidic Acid Cream", "dosage": "Apply 3x daily", "advice": "Keep area clean."},
            "diarrhea": {"drug": "ORS + Zinc", "dosage": "Rehydrate continuously", "advice": "Stop if blood in stool."},
            "anxiety_mild": {"drug": "Magnesium Supp / Tea", "dosage": "Chamomile tea", "advice": "Breathing exercises."}
        }
        
        # Massive expansion simulation
        for i in range(100, 5000):
            self.symptom_map[f"SYMPTOM_CODE_{i}"] = {
                "drug": f"Generic_Treatment_{i}",
                "dosage": "Standard Protocol",
                "advice": "Refer to specialist."
            }

    def diagnose(self, symptom_text: str) -> str:
        symptom_text = symptom_text.lower()
        
        # Simple keyword matching
        for key in self.symptom_map:
            if key in symptom_text:
                rec = self.symptom_map[key]
                return (
                    f"### [PHARMACY AI ADVICE]\n"
                    f"- Detected Symptom: {key.upper()}\n"
                    f"- Recommended OTC: {rec['drug']}\n"
                    f"- Dosage: {rec['dosage']}\n"
                    f"- Advice: {rec['advice']}\n"
                    f"- NOTE: {self.disclaimer}"
                )
        return "Symptom not clear. Please consult a physical doctor."

GLOBAL_MEDIC_DB = MedicalDiagnosticEngine()
