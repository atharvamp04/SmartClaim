import random
from datetime import datetime

# ============================================================
# 🪪 DRIVING LICENSE VERIFICATION (SIMULATED)
# ============================================================
def verify_dl(dl_number=None, expiry_date=None):
    """
    Simulate driving license verification.
    Checks expiry and pattern to estimate reliability.
    """
    if not dl_number:
        return {'valid': False, 'dl_score': 0.4, 'reason': 'Missing DL number'}

    # Basic validation pattern (2 letters + digits)
    valid_format = dl_number[:2].isalpha() and dl_number[2:].isdigit()

    # Expiry check
    expired = False
    if expiry_date:
        try:
            exp = datetime.strptime(str(expiry_date)[:10], "%Y-%m-%d")
            expired = exp < datetime.now()
        except Exception:
            expired = True

    # Compute DL score
    if valid_format and not expired:
        score = round(random.uniform(0.8, 1.0), 2)
    elif valid_format and expired:
        score = round(random.uniform(0.3, 0.6), 2)
    else:
        score = round(random.uniform(0.1, 0.4), 2)

    return {
        'valid': valid_format,
        'expired': expired,
        'dl_score': score,
        'reason': 'OK' if valid_format else 'Invalid DL number'
    }


# ============================================================
# 🚗 RTO / VEHICLE REGISTRATION VERIFICATION (SIMULATED)
# ============================================================
def verify_rto(reg_no=None, make=None, year=None):
    """
    Simulate RTO verification using reg number & year.
    """
    if not reg_no:
        return {'valid': False, 'rto_score': 0.4, 'reason': 'Missing RC number'}

    valid_pattern = len(reg_no) >= 8 and reg_no[:2].isalpha()
    year_valid = (1990 <= int(year or 0) <= datetime.now().year)

    if valid_pattern and year_valid:
        score = round(random.uniform(0.8, 1.0), 2)
    elif valid_pattern:
        score = round(random.uniform(0.5, 0.7), 2)
    else:
        score = round(random.uniform(0.1, 0.4), 2)

    return {
        'valid': valid_pattern,
        'year_valid': year_valid,
        'rto_score': score,
        'reason': 'OK' if valid_pattern else 'Invalid RC format'
    }


# ============================================================
# 🚔 FIR / POLICE RECORD VERIFICATION (SIMULATED)
# ============================================================
def verify_fir(fir_no=None):
    """
    Simulate checking if an FIR is filed for the claim.
    """
    if not fir_no:
        return {'exists': False, 'fir_score': 0.8, 'reason': 'No FIR filed'}

    # Random simulation: 70% genuine, 30% fraud related
    fraud_related = random.random() < 0.3
    score = round(random.uniform(0.2, 0.5), 2) if fraud_related else round(random.uniform(0.8, 1.0), 2)

    return {
        'exists': True,
        'fraud_related': fraud_related,
        'fir_score': score,
        'reason': 'Suspicious FIR' if fraud_related else 'Clean FIR'
    }


# ============================================================
# 🧮 AGGREGATION
# ============================================================
def aggregate_verification(dl_info, rto_info, fir_info):
    """
    Combine all three verification checks into one reliability score.
    """
    scores = [
        dl_info.get('dl_score', 0.5),
        rto_info.get('rto_score', 0.5),
        fir_info.get('fir_score', 0.5)
    ]
    reliability = sum(scores) / len(scores)
    return round(reliability, 2)
