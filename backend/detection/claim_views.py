# detection/claim_views.py
"""
API Views for Claim Management
Handles claim retrieval, status updates, and admin operations
"""

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework import status
from django.db.models import Q

from .claim_handler import ClaimDatabaseHandler
from .models import Claim, Policyholder


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_all_claims(request):
    """
    Get all claims with optional filtering
    Query params:
        - status: Filter by status (Pending, Verified, Fraud, Rejected)
        - policyholder: Filter by policyholder username
        - risk_level: Filter by risk level (LOW, MEDIUM, HIGH)
    """
    try:
        # Get query parameters
        status_filter = request.GET.get('status')
        policyholder_filter = request.GET.get('policyholder')
        risk_level_filter = request.GET.get('risk_level')
        
        # Get claims
        claims = ClaimDatabaseHandler.get_all_claims(status_filter=status_filter)
        
        # Apply additional filters
        if policyholder_filter:
            claims = claims.filter(policyholder__username=policyholder_filter)
        
        if risk_level_filter:
            claims = claims.filter(risk_level=risk_level_filter)
        
        # Serialize claims
        claims_data = [
            ClaimDatabaseHandler.serialize_claim(claim, include_images=False, include_history=False)
            for claim in claims
        ]
        
        return Response({
            'count': len(claims_data),
            'claims': claims_data
        })
        
    except Exception as e:
        return Response({
            'error': f'Failed to retrieve claims: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_claim_detail(request, claim_id):
    """Get detailed information for a specific claim"""
    try:
        claim = ClaimDatabaseHandler.get_claim_by_id(claim_id)
        
        if not claim:
            return Response({
                'error': 'Claim not found'
            }, status=status.HTTP_404_NOT_FOUND)
        
        # Serialize with full details
        claim_data = ClaimDatabaseHandler.serialize_claim(
            claim, 
            include_images=True, 
            include_history=True
        )
        
        return Response(claim_data)
        
    except Exception as e:
        return Response({
            'error': f'Failed to retrieve claim: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_claim_by_number(request, claim_number):
    """Get claim by claim number"""
    try:
        claim = ClaimDatabaseHandler.get_claim_by_number(claim_number)
        
        if not claim:
            return Response({
                'error': 'Claim not found'
            }, status=status.HTTP_404_NOT_FOUND)
        
        claim_data = ClaimDatabaseHandler.serialize_claim(
            claim, 
            include_images=True, 
            include_history=True
        )
        
        return Response(claim_data)
        
    except Exception as e:
        return Response({
            'error': f'Failed to retrieve claim: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_pending_claims(request):
    """Get all pending claims"""
    try:
        claims = ClaimDatabaseHandler.get_all_claims(status_filter='Pending')
        
        claims_data = [
            ClaimDatabaseHandler.serialize_claim(claim, include_images=False, include_history=False)
            for claim in claims
        ]
        
        return Response({
            'count': len(claims_data),
            'claims': claims_data
        })
        
    except Exception as e:
        return Response({
            'error': f'Failed to retrieve pending claims: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_verified_claims(request):
    """Get all verified/claimed claims"""
    try:
        claims = ClaimDatabaseHandler.get_all_claims(status_filter='Verified')
        
        claims_data = [
            ClaimDatabaseHandler.serialize_claim(claim, include_images=False, include_history=False)
            for claim in claims
        ]
        
        return Response({
            'count': len(claims_data),
            'claims': claims_data
        })
        
    except Exception as e:
        return Response({
            'error': f'Failed to retrieve verified claims: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_fraud_claims(request):
    """Get all fraud-detected claims"""
    try:
        claims = ClaimDatabaseHandler.get_fraud_detected_claims()
        
        claims_data = [
            ClaimDatabaseHandler.serialize_claim(claim, include_images=False, include_history=False)
            for claim in claims
        ]
        
        return Response({
            'count': len(claims_data),
            'claims': claims_data
        })
        
    except Exception as e:
        return Response({
            'error': f'Failed to retrieve fraud claims: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_high_risk_claims(request):
    """Get all high-risk claims requiring review"""
    try:
        claims = ClaimDatabaseHandler.get_high_risk_claims()
        
        claims_data = [
            ClaimDatabaseHandler.serialize_claim(claim, include_images=False, include_history=False)
            for claim in claims
        ]
        
        return Response({
            'count': len(claims_data),
            'claims': claims_data
        })
        
    except Exception as e:
        return Response({
            'error': f'Failed to retrieve high-risk claims: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def update_claim_status(request, claim_id):
    """
    Update claim status
    Body: {
        "status": "Pending|Verified|Fraud|Rejected",
        "notes": "Optional notes about the status change"
    }
    """
    try:
        new_status = request.data.get('status')
        notes = request.data.get('notes')
        admin_user = request.user.username
        
        if not new_status:
            return Response({
                'error': 'Status is required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        if new_status not in ['Pending', 'Verified', 'Fraud', 'Rejected']:
            return Response({
                'error': 'Invalid status. Must be Pending, Verified, Fraud, or Rejected'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        claim, updated = ClaimDatabaseHandler.update_claim_status(
            claim_id, new_status, admin_user, notes
        )
        
        if not claim:
            return Response({
                'error': 'Claim not found'
            }, status=status.HTTP_404_NOT_FOUND)
        
        if not updated:
            return Response({
                'message': 'No change - claim already has this status',
                'claim': ClaimDatabaseHandler.serialize_claim(claim, include_images=False, include_history=True)
            })
        
        return Response({
            'message': f'Claim status updated to {new_status}',
            'claim': ClaimDatabaseHandler.serialize_claim(claim, include_images=False, include_history=True)
        })
        
    except Exception as e:
        return Response({
            'error': f'Failed to update claim status: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def add_admin_review(request, claim_id):
    """
    Add admin review to a claim
    Body: {
        "notes": "Admin review notes",
        "status": "Optional new status"
    }
    """
    try:
        notes = request.data.get('notes')
        new_status = request.data.get('status')
        admin_user = request.user.username
        
        if not notes:
            return Response({
                'error': 'Review notes are required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        claim = ClaimDatabaseHandler.add_admin_review(
            claim_id, admin_user, notes, new_status
        )
        
        if not claim:
            return Response({
                'error': 'Claim not found'
            }, status=status.HTTP_404_NOT_FOUND)
        
        return Response({
            'message': 'Review added successfully',
            'claim': ClaimDatabaseHandler.serialize_claim(claim, include_images=False, include_history=True)
        })
        
    except Exception as e:
        return Response({
            'error': f'Failed to add review: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_claim_history(request, claim_id):
    """Get complete history for a claim"""
    try:
        history = ClaimDatabaseHandler.get_claim_history(claim_id)
        
        history_data = [
            {
                'action': hist.action,
                'old_status': hist.old_status,
                'new_status': hist.new_status,
                'performed_by': hist.performed_by,
                'notes': hist.notes,
                'timestamp': hist.timestamp.isoformat()
            }
            for hist in history
        ]
        
        return Response({
            'count': len(history_data),
            'history': history_data
        })
        
    except Exception as e:
        return Response({
            'error': f'Failed to retrieve claim history: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_claim_statistics(request):
    """Get overall claim statistics"""
    try:
        stats = ClaimDatabaseHandler.get_claim_statistics()
        
        return Response({
            'statistics': stats,
            'fraud_rate': (stats['fraud_detected_count'] / stats['total_claims'] * 100) if stats['total_claims'] > 0 else 0,
            'high_risk_rate': (stats['high_risk_count'] / stats['total_claims'] * 100) if stats['total_claims'] > 0 else 0
        })
        
    except Exception as e:
        return Response({
            'error': f'Failed to retrieve statistics: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_policyholder_claims(request, username):
    """Get all claims for a specific policyholder"""
    try:
        history = ClaimDatabaseHandler.get_policyholder_claim_history(username)
        
        if not history:
            return Response({
                'error': 'Policyholder not found'
            }, status=status.HTTP_404_NOT_FOUND)
        
        # Serialize claims
        claims_data = [
            ClaimDatabaseHandler.serialize_claim(claim, include_images=False, include_history=False)
            for claim in history['claims']
        ]
        
        return Response({
            'policyholder': {
                'username': history['policyholder'].username,
                'email': history['policyholder'].email
            },
            'summary': {
                'total_claims': history['total_claims'],
                'pending_claims': history['pending_claims'],
                'verified_claims': history['verified_claims'],
                'fraud_claims': history['fraud_claims'],
                'rejected_claims': history['rejected_claims']
            },
            'claims': claims_data
        })
        
    except Exception as e:
        return Response({
            'error': f'Failed to retrieve policyholder claims: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_claim(request, claim_id):
    """Delete a claim (admin only)"""
    try:
        claim = ClaimDatabaseHandler.get_claim_by_id(claim_id)
        
        if not claim:
            return Response({
                'error': 'Claim not found'
            }, status=status.HTTP_404_NOT_FOUND)
        
        # Store claim number before deletion
        claim_number = claim.claim_number
        
        # Delete claim (cascades to images and history)
        claim.delete()
        
        return Response({
            'message': f'Claim {claim_number} deleted successfully'
        })
        
    except Exception as e:
        return Response({
            'error': f'Failed to delete claim: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def search_claims(request):
    """
    Search claims by various criteria
    Query params:
        - q: Search query (claim number, policyholder username, description)
        - status: Filter by status
        - risk_level: Filter by risk level
        - fraud_detected: Filter by fraud detection (true/false)
        - min_amount: Minimum claim amount
        - max_amount: Maximum claim amount
        - date_from: Start date (YYYY-MM-DD)
        - date_to: End date (YYYY-MM-DD)
    """
    try:
        query = request.GET.get('q', '')
        status_filter = request.GET.get('status')
        risk_level = request.GET.get('risk_level')
        fraud_detected = request.GET.get('fraud_detected')
        min_amount = request.GET.get('min_amount')
        max_amount = request.GET.get('max_amount')
        date_from = request.GET.get('date_from')
        date_to = request.GET.get('date_to')
        
        # Start with all claims
        claims = Claim.objects.select_related('policyholder').prefetch_related('images')
        
        # Apply text search
        if query:
            claims = claims.filter(
                Q(claim_number__icontains=query) |
                Q(policyholder__username__icontains=query) |
                Q(claim_description__icontains=query) |
                Q(dl_number__icontains=query) |
                Q(vehicle_reg_no__icontains=query) |
                Q(fir_number__icontains=query)
            )
        
        # Apply filters
        if status_filter:
            claims = claims.filter(status=status_filter)
        
        if risk_level:
            claims = claims.filter(risk_level=risk_level)
        
        if fraud_detected is not None:
            is_fraud = fraud_detected.lower() == 'true'
            claims = claims.filter(fraud_detected=is_fraud)
        
        if min_amount:
            claims = claims.filter(claim_amount__gte=float(min_amount))
        
        if max_amount:
            claims = claims.filter(claim_amount__lte=float(max_amount))
        
        if date_from:
            claims = claims.filter(submitted_at__gte=date_from)
        
        if date_to:
            claims = claims.filter(submitted_at__lte=date_to)
        
        # Order by newest first
        claims = claims.order_by('-submitted_at')
        
        # Serialize results
        claims_data = [
            ClaimDatabaseHandler.serialize_claim(claim, include_images=False, include_history=False)
            for claim in claims
        ]
        
        return Response({
            'count': len(claims_data),
            'query': query,
            'filters_applied': {
                'status': status_filter,
                'risk_level': risk_level,
                'fraud_detected': fraud_detected,
                'amount_range': f"{min_amount or 'any'} - {max_amount or 'any'}",
                'date_range': f"{date_from or 'any'} - {date_to or 'any'}"
            },
            'claims': claims_data
        })
        
    except Exception as e:
        return Response({
            'error': f'Search failed: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([AllowAny])
def claim_api_info(request):
    """API information and available endpoints"""
    return Response({
        'api_version': '1.0',
        'description': 'Claim Management API',
        'endpoints': {
            'GET /api/claims/': 'Get all claims (with filters)',
            'GET /api/claims/<id>/': 'Get claim detail',
            'GET /api/claims/number/<claim_number>/': 'Get claim by number',
            'GET /api/claims/pending/': 'Get pending claims',
            'GET /api/claims/verified/': 'Get verified claims',
            'GET /api/claims/fraud/': 'Get fraud claims',
            'GET /api/claims/high-risk/': 'Get high-risk claims',
            'POST /api/claims/<id>/status/': 'Update claim status',
            'POST /api/claims/<id>/review/': 'Add admin review',
            'GET /api/claims/<id>/history/': 'Get claim history',
            'GET /api/claims/statistics/': 'Get claim statistics',
            'GET /api/claims/policyholder/<username>/': 'Get policyholder claims',
            'GET /api/claims/search/': 'Search claims',
            'DELETE /api/claims/<id>/': 'Delete claim'
        },
        'authentication': 'JWT Token required (except /info/)',
        'note': 'Most endpoints require IsAuthenticated permission'
    })