"""
Real-world Document Verification APIs
Integrates with actual Indian government and third-party services
"""

import requests
import logging
from decouple import config
from datetime import datetime

logger = logging.getLogger(__name__)

# API Configuration
USE_MOCK = config('USE_MOCK_APIS', default=True, cast=bool)

# Real API Keys (Add these to your .env file)
SUREPASS_API_KEY = config('SUREPASS_API_KEY', default='')
SIGNZY_API_KEY = config('SIGNZY_API_KEY', default='')
PARIVAHAN_API_KEY = config('PARIVAHAN_API_KEY', default='')

# API Endpoints
SUREPASS_BASE_URL = "https://kyc-api.aadhaarkyc.io/api/v1"
SIGNZY_BASE_URL = "https://api.signzy.tech/api/v2"
PARIVAHAN_BASE_URL = "https://www.parivahan.gov.in/rcdlstatus"


def verify_driving_license_real(dl_number, dob=None):
    """
    Verify Driving License using Parivahan/Surepass API
    
    Provider Options:
    1. Surepass (Paid) - https://surepass.io/
    2. Signzy (Paid) - https://signzy.com/
    3. Parivahan Sewa (Official but complex)
    """
    
    if not SUREPASS_API_KEY:
        logger.warning("SUREPASS_API_KEY not configured, using mock")
        return verify_driving_license_mock(dl_number, dob)
    
    try:
        headers = {
            'Authorization': f'Bearer {SUREPASS_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'id_number': dl_number
        }
        
        if dob:
            payload['dob'] = dob
        
        response = requests.post(
            f"{SUREPASS_BASE_URL}/driving-license/driving-license",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('success'):
                dl_data = data.get('data', {})
                
                # Parse expiry date
                expiry_date = dl_data.get('validity', {}).get('non_transport', '')
                is_valid = True
                
                if expiry_date:
                    try:
                        expiry = datetime.strptime(expiry_date, '%d-%m-%Y')
                        is_valid = expiry > datetime.now()
                    except:
                        pass
                
                return {
                    'verified': True,
                    'fraud_indicator': 'LOW',
                    'message': 'License verified successfully',
                    'license_details': {
                        'dl_number': dl_data.get('dl_number'),
                        'holder_name': dl_data.get('name'),
                        'status': 'ACTIVE' if is_valid else 'EXPIRED',
                        'license_valid': is_valid,
                        'expiry_date': expiry_date,
                        'license_class': dl_data.get('vehicle_class', []),
                        'issue_date': dl_data.get('issue_date'),
                        'rto': dl_data.get('rto_name'),
                        'address': dl_data.get('permanent_address')
                    },
                    'fraud_checks': {
                        'expired': not is_valid,
                        'suspended': False,
                        'fake': False
                    },
                    'source': 'surepass_api'
                }
            else:
                return {
                    'verified': False,
                    'fraud_indicator': 'HIGH',
                    'message': data.get('message', 'License verification failed'),
                    'source': 'surepass_api'
                }
        
        else:
            logger.error(f"Surepass API error: {response.status_code}")
            return verify_driving_license_mock(dl_number, dob)
            
    except requests.Timeout:
        logger.error("Surepass API timeout")
        return verify_driving_license_mock(dl_number, dob)
    
    except Exception as e:
        logger.error(f"DL verification error: {str(e)}")
        return verify_driving_license_mock(dl_number, dob)


def verify_vehicle_registration_real(registration_number, chassis_number=None, engine_number=None):
    """
    Verify Vehicle RC using Parivahan/Surepass API
    
    Provider Options:
    1. Surepass (Paid) - https://surepass.io/
    2. Signzy (Paid) - https://signzy.com/
    3. Vahan API (Official but needs authorization)
    """
    
    if not SUREPASS_API_KEY:
        logger.warning("SUREPASS_API_KEY not configured, using mock")
        return verify_vehicle_registration_mock(registration_number)
    
    try:
        headers = {
            'Authorization': f'Bearer {SUREPASS_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'id_number': registration_number.replace(' ', '').upper()
        }
        
        response = requests.post(
            f"{SUREPASS_BASE_URL}/rc/rc",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('success'):
                rc_data = data.get('data', {})
                
                # Parse insurance validity
                insurance_expiry = rc_data.get('insurance_validity', '')
                insurance_valid = True
                
                if insurance_expiry:
                    try:
                        expiry = datetime.strptime(insurance_expiry, '%d-%b-%Y')
                        insurance_valid = expiry > datetime.now()
                    except:
                        pass
                
                # Check if vehicle is blacklisted
                blacklist_status = rc_data.get('blacklist_status', 'NA')
                is_blacklisted = blacklist_status != 'NA' and blacklist_status != 'No'
                
                return {
                    'verified': True,
                    'fraud_indicator': 'HIGH' if is_blacklisted else 'LOW',
                    'message': 'Vehicle verified successfully',
                    'vehicle_details': {
                        'registration_number': rc_data.get('registration_number'),
                        'owner_name': rc_data.get('owner_name'),
                        'vehicle_class': rc_data.get('vehicle_class'),
                        'status': 'BLACKLISTED' if is_blacklisted else 'ACTIVE',
                        'insurance_valid': insurance_valid,
                        'insurance_expiry': insurance_expiry,
                        'insurance_company': rc_data.get('insurance_company'),
                        'vehicle_make': rc_data.get('maker_model'),
                        'chassis_number': rc_data.get('chassis_number'),
                        'engine_number': rc_data.get('engine_number'),
                        'registration_date': rc_data.get('registration_date'),
                        'rto': rc_data.get('rto')
                    },
                    'fraud_checks': {
                        'stolen': False,  # Would need separate API
                        'duplicate': False,
                        'blacklisted': is_blacklisted,
                        'chassis_match': chassis_number == rc_data.get('chassis_number') if chassis_number else None,
                        'engine_match': engine_number == rc_data.get('engine_number') if engine_number else None
                    },
                    'source': 'surepass_api'
                }
            else:
                return {
                    'verified': False,
                    'fraud_indicator': 'HIGH',
                    'message': data.get('message', 'Vehicle verification failed'),
                    'source': 'surepass_api'
                }
        
        else:
            logger.error(f"Surepass API error: {response.status_code}")
            return verify_vehicle_registration_mock(registration_number)
            
    except requests.Timeout:
        logger.error("Surepass API timeout")
        return verify_vehicle_registration_mock(registration_number)
    
    except Exception as e:
        logger.error(f"Vehicle verification error: {str(e)}")
        return verify_vehicle_registration_mock(registration_number)


def verify_police_report_real(fir_number, accident_date=None, location=None):
    """
    Verify Police FIR
    
    Note: There's no public API for FIR verification in India
    Options:
    1. Manual verification through police station
    2. State-specific portals (limited access)
    3. Use mock/manual verification for now
    
    For production: Integrate with state police portals or use manual verification
    """
    
    # For now, we'll use enhanced mock verification
    return verify_police_report_mock(fir_number, accident_date, location)


# Mock functions (fallback)
def verify_driving_license_mock(dl_number, dob=None):
    """Mock DL verification"""
    if not dl_number or len(dl_number) < 15:
        return {
            'verified': False,
            'fraud_indicator': 'HIGH',
            'message': 'Invalid license number',
            'source': 'mock'
        }
    
    if dl_number[:2].isalpha():
        return {
            'verified': True,
            'fraud_indicator': 'LOW',
            'message': 'License verified successfully (MOCK)',
            'license_details': {
                'dl_number': dl_number,
                'holder_name': 'Mock User',
                'status': 'ACTIVE',
                'license_valid': True,
                'expiry_date': '2030-01-01',
                'license_class': ['LMV', 'MCWG']
            },
            'fraud_checks': {
                'expired': False,
                'suspended': False,
                'fake': False
            },
            'source': 'mock'
        }
    else:
        return {
            'verified': False,
            'fraud_indicator': 'HIGH',
            'message': 'Invalid license format',
            'source': 'mock'
        }


def verify_vehicle_registration_mock(registration_number):
    """Mock vehicle verification"""
    if not registration_number or len(registration_number) < 10:
        return {
            'verified': False,
            'fraud_indicator': 'HIGH',
            'message': 'Invalid registration number',
            'source': 'mock'
        }
    
    if registration_number[:2].isalpha() and registration_number[2:4].isdigit():
        return {
            'verified': True,
            'fraud_indicator': 'LOW',
            'message': 'Vehicle verified successfully (MOCK)',
            'vehicle_details': {
                'registration_number': registration_number,
                'owner_name': 'Mock Owner',
                'vehicle_class': 'LMV',
                'status': 'ACTIVE',
                'insurance_valid': True,
                'insurance_expiry': '2026-01-01'
            },
            'fraud_checks': {
                'stolen': False,
                'duplicate': False,
                'blacklisted': False
            },
            'source': 'mock'
        }
    else:
        return {
            'verified': False,
            'fraud_indicator': 'HIGH',
            'message': 'Invalid vehicle registration format',
            'source': 'mock'
        }


def verify_police_report_mock(fir_number, accident_date=None, location=None):
    """Mock FIR verification"""
    if not fir_number or len(fir_number) < 10:
        return {
            'verified': False,
            'fraud_indicator': 'HIGH',
            'message': 'Invalid FIR number format',
            'source': 'mock'
        }
    
    if fir_number[:3].isdigit():
        return {
            'verified': True,
            'fraud_indicator': 'LOW',
            'message': 'FIR verified successfully (MOCK)',
            'fir_details': {
                'number': fir_number,
                'status': 'REGISTERED',
                'police_station': location or 'Mock Police Station'
            },
            'source': 'mock'
        }
    else:
        return {
            'verified': False,
            'fraud_indicator': 'HIGH',
            'message': 'FIR not found in records',
            'source': 'mock'
        }


# Main API switch functions
def verify_driving_license(dl_number, dob=None, name=None):
    """Main DL verification function"""
    if USE_MOCK:
        return verify_driving_license_mock(dl_number, dob)
    else:
        return verify_driving_license_real(dl_number, dob)


def verify_vehicle_registration(registration_number, chassis_number=None, engine_number=None):
    """Main vehicle verification function"""
    if USE_MOCK:
        return verify_vehicle_registration_mock(registration_number)
    else:
        return verify_vehicle_registration_real(registration_number, chassis_number, engine_number)


def verify_police_report(fir_number, accident_date=None, location=None):
    """Main FIR verification function"""
    return verify_police_report_real(fir_number, accident_date, location)


import random

def verify_dl(dl_number=None, expiry_date=None):
    """
    Simulated Driving License verification.
    In production: integrate Parivahan or MoRTH API.
    """
    if not dl_number:
        return {"valid": False, "dl_score": 0.0, "message": "DL number missing"}

    expired = False
    if expiry_date:
        try:
            from datetime import datetime
            exp = datetime.fromisoformat(str(expiry_date))
            expired = exp < datetime.now()
        except Exception:
            pass

    dl_score = 0.9 if not expired else 0.3
    return {"valid": not expired, "dl_score": dl_score, "message": "DL verified" if not expired else "DL expired"}


def verify_rto(reg_no=None, make=None, year=None):
    """
    Simulated RTO API verification.
    Checks if reg_no format, make, and year are consistent.
    """
    if not reg_no:
        return {"valid": False, "rto_score": 0.0, "message": "Registration missing"}

    valid_make = make and make.lower() in ["maruti", "honda", "hyundai", "bmw", "audi", "tata", "mahindra"]
    valid_year = year and 1990 < int(year) <= 2025
    rto_score = 0.8 if (valid_make and valid_year) else 0.4

    return {
        "valid": valid_make and valid_year,
        "rto_score": rto_score,
        "message": "RTO verified" if valid_make and valid_year else "Mismatch detected"
    }


def verify_fir(fir_no=None):
    """
    Simulated FIR verification API.
    In production: police record integration (state-wise API).
    """
    if not fir_no:
        return {"exists": False, "fir_score": 0.5, "message": "No FIR number provided"}
    fake_fraud_case = fir_no.lower().startswith("fraud")
    fir_score = 0.9 if not fake_fraud_case else 0.2
    return {"exists": True, "fir_score": fir_score, "message": "FIR verified"}


def aggregate_verification(dl_info, rto_info, fir_info):
    """
    Combines DL, RTO, and FIR reliability using tuned weights.
    Higher = more reliable verification.
    """
    DL_WEIGHT = 0.40
    RTO_WEIGHT = 0.35
    FIR_WEIGHT = 0.25

    reliability = (
        (DL_WEIGHT * dl_info.get("dl_score", 0)) +
        (RTO_WEIGHT * rto_info.get("rto_score", 0)) +
        (FIR_WEIGHT * fir_info.get("fir_score", 0))
    )
    return round(reliability, 3)
