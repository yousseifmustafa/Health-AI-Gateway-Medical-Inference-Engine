from pydantic import BaseModel, Field
from typing import List, Optional

class MedicationEntry(BaseModel):
    """A single medication line extracted from the prescription."""
    name:      Optional[str] = Field(None, description="The medication name exactly as written (trade or generic). Use null if illegible.")
    dosage:    Optional[str] = Field(None, description="Dosage per administration (e.g. '500mg', '1 tablet'). Use null if illegible.")
    frequency: Optional[str] = Field(None, description="How often to take (e.g. 'twice daily', 'مرتين يومياً'). Use null if illegible.")
    duration:  Optional[str] = Field(None, description="Course length if specified (e.g. '7 days').")
    notes:     Optional[str] = Field(None, description="Any special instruction for this drug (e.g. 'take with food').")

class PrescriptionResponse(BaseModel):
    """Structured data extracted from a handwritten or printed medical prescription."""
    patient_name:      Optional[str]             = Field(None, description="Patient name if legible.")
    patient_date:      Optional[str]             = Field(None, description="Date on the prescription if present.")
    doctor_name:       Optional[str]             = Field(None, description="Prescribing doctor name if legible.")
    medications:       List[MedicationEntry]     = Field(default_factory=list, description="All medications listed.")
    general_notes:     Optional[str]             = Field(None, description="Any free-text doctor instructions not tied to a specific drug.")
    confidence:        str                       = Field(default="MEDIUM", description="Overall OCR confidence: 'HIGH', 'MEDIUM', or 'LOW'.")
    unreadable_parts:  Optional[str]             = Field(None, description="Describe any parts of the image that were too unclear to parse.")

class MedicineBoxResponse(BaseModel):
    """Structured data extracted from a medicine box / packaging image."""
    trade_name:          str            = Field(description="Brand/commercial name printed on the box.")
    generic_name:        Optional[str]  = Field(None, description="INN / active ingredient name.")
    active_ingredients:  List[str]      = Field(default_factory=list, description="List of active chemical compounds with concentrations.")
    concentration:       Optional[str]  = Field(None, description="Strength of the formulation (e.g. '500mg/5ml').")
    dosage_form:         Optional[str]  = Field(None, description="Form type (e.g. tablet, syrup, injection).")
    indications:         List[str]      = Field(default_factory=list, description="Stated medical uses / indications.")
    contraindications:   List[str]      = Field(default_factory=list, description="Stated warnings or contraindications.")
    manufacturer:        Optional[str]  = Field(None, description="Manufacturing company name.")
    storage_conditions:  Optional[str]  = Field(None, description="Storage instructions if visible.")
    expiry_date:         Optional[str]  = Field(None, description="Expiry date if legible.")
