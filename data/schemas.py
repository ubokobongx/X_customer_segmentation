# data/schemas.py
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from enum import Enum

class RiskLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

class ValueLevel(str, Enum):
    HIGH = "High Value"
    LOW = "Low Value"

class CustomerSegment(str, Enum):
    LOW_HIGH = "Low Risk - High Value"
    LOW_LOW = "Low Risk - Low Value"
    MEDIUM_HIGH = "Medium Risk - High Value"
    MEDIUM_LOW = "Medium Risk - Low Value"
    HIGH = "High Risk"

class RawCustomerData(BaseModel):
    """Schema for raw customer data from Redshift."""
    customer_id: int = Field(..., gt=0)
    loan_count: int = Field(..., ge=0)
    total_amount_overdue: float = Field(..., ge=0)
    maturity_dpd: int = Field(..., ge=0)
    total_missed_installment: int = Field(..., ge=0)
    has_14plus_dpd: int = Field(0, ge=0, le=1)
    count_14plus_dpd: int = Field(..., ge=0)
    tenor_in_months: float = Field(..., gt=0)  # CHANGED: float instead of int
    ontime_repayment_rate: float = Field(..., ge=0, le=100)
    age: int = Field(..., ge=18, le=100)
    gender: str
    marital_status: Optional[str] = None  # CHANGED: Optional
    state: Optional[str] = None  # CHANGED: Optional
    location: Optional[str] = None  # CHANGED: Optional
    purpose: Optional[str] = None  # CHANGED: Optional
    employment_status: Optional[str] = None  # CHANGED: Optional
    dw_channel_key: Optional[str] = None  # CHANGED: Optional
    total_loan_amount: float = Field(..., gt=0)
    income: float = Field(..., ge=0)
    
    # Add validators to handle data cleaning
    @validator('tenor_in_months', pre=True)
    def clean_tenor(cls, v):
        """Convert tenor to proper float, handle None."""
        if v is None:
            return 1.0  # Default minimum tenor
        try:
            return float(v)
        except (ValueError, TypeError):
            return 1.0
    
    @validator('marital_status', 'state', 'location', 'purpose', 'employment_status', 'dw_channel_key', pre=True)
    def clean_string_fields(cls, v):
        """Convert None to empty string for string fields."""
        if v is None:
            return ''
        return str(v)
    
    @validator('gender', pre=True)
    def clean_gender(cls, v):
        """Clean gender field."""
        if v is None:
            return 'Unknown'
        v = str(v).strip().title()
        if v not in ['Male', 'Female', 'Other', 'Unknown']:
            return 'Unknown'
        return v
    
    class Config:
        extra = "ignore"  # Ignore extra fields

class ProcessedCustomerData(RawCustomerData):
    """Schema for processed customer data."""
    income_bracket: Optional[str] = None
    age_category: Optional[str] = None
    missed_payment_ratio: Optional[float] = None
    overdue_utilization: Optional[float] = None
    monthly_loan_volume: Optional[float] = None
    repayment_efficiency: Optional[float] = None
    segment: Optional[CustomerSegment] = None
    
    @validator('missed_payment_ratio', 'overdue_utilization', 'repayment_efficiency')
    def validate_ratios(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError(f"Ratio must be between 0 and 1, got {v}")
        return v

class SegmentProfile(BaseModel):
    """Schema for segment profiles."""
    segment: CustomerSegment
    combination_id: str
    combination: str
    observed_proportion: float
    expected_proportion: float
    odds_ratio: float
    count: int
    coverage: float
    
    class Config:
        extra = "forbid"