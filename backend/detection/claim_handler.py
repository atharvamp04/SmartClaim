# detection/claim_handler.py
"""
Claim Database Handler
Manages claim submission, retrieval, status updates, and history tracking
"""

import json
from datetime import datetime
from django.utils import timezone
from django.db import transaction
from django.core.files.base import ContentFile
from PIL import Image
import io
import base64
import json
import numpy as np
from .models import Claim, ClaimImage, ClaimHistory, Policyholder

def make_json_safe(obj):
    """
    Recursively convert non-serializable types (numpy, bool, etc.)
    into JSON-safe Python types.
    """
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(i) for i in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, bytes):
        return obj.decode('utf-8', errors='ignore')
    else:
        return obj

class ClaimDatabaseHandler:
    """Handles database operations for claims."""

    @staticmethod
    @transaction.atomic
    def create_claim(policyholder, claim_data, fraud_detection_result, image_files=None):
        """
        Create a new claim with all fraud detection data
        """
        try:
            # === Extract fraud details ===
            fraud_detected = fraud_detection_result.get('fraud_detected', False)
            confidence_score = fraud_detection_result.get('confidence', 0) * 100
            risk_level = fraud_detection_result.get('risk_level', 'MEDIUM')

            # === Get detailed analysis ===
            detailed_calc = fraud_detection_result.get('detailed_calculations', {})
            tabular_analysis = detailed_calc.get('tabular_analysis', {})
            image_analysis = detailed_calc.get('image_analysis', {})
            fusion_analysis = detailed_calc.get('fusion_analysis', {})

            # === Verification details ===
            multi_image_analysis = fraud_detection_result.get('multi_image_analysis', {})
            aggregated_metrics = multi_image_analysis.get('aggregated_metrics', {})
            fusion_analysis = fraud_detection_result.get('fusion_analysis', {})
            verification_details = fusion_analysis.get('verification_details', {})

            dl_score = verification_details.get('dl', {}).get('dl_score', 0)
            rto_score = verification_details.get('rto', {}).get('rto_score', 0)
            fir_score = verification_details.get('fir', {}).get('fir_score', 0)
            combined_reliability = fusion_analysis.get('input_probabilities', {}).get('verification_reliability', 0)

            # === Determine claim status ===
            if fraud_detected:
                status = 'Fraud'
            elif risk_level == 'HIGH':
                status = 'Pending'
            else:
                status = 'Pending'

            fraud_detection_result = make_json_safe(fraud_detection_result)

            # === Create main claim ===
            claim = Claim.objects.create(
                policyholder=policyholder,
                claim_description=claim_data.get('claim_description', ''),
                accident_date=claim_data.get('accident_date'),
                claim_amount=claim_data.get('claim_amount', 0),
                dl_number=claim_data.get('dl_number', ''),
                vehicle_reg_no=claim_data.get('vehicle_reg_no', ''),
                fir_number=claim_data.get('fir_number', ''),
                total_images_submitted=fraud_detection_result.get('total_images_submitted', 1),
                image_paths=[],
                fraud_detected=fraud_detected,
                confidence_score=confidence_score,
                risk_level=risk_level,
                status=status,
                tabular_fraud_probability=tabular_analysis.get('probabilities', {}).get('fraud', 0) * 100,
                image_fraud_probability=aggregated_metrics.get(
                    'final_image_fraud_probability',
                    image_analysis.get('image_fraud_probability', 0)
                ) * 100,
                fusion_score=fusion_analysis.get('final_confidence', 0) * 100,
                overall_damage_severity=aggregated_metrics.get('severity_analysis', {}).get('overall_severity', 'LOW'),
                total_damage_areas=aggregated_metrics.get('damage_summary', {}).get('total_detections_all_images', 0),
                average_damage_percentage=aggregated_metrics.get('damage_summary', {}).get('avg_damage_percentage', 0),
                max_fraud_image_index=aggregated_metrics.get('aggregation_components', {}).get('max_fraud_image_index'),
                max_fraud_probability=aggregated_metrics.get('aggregation_components', {}).get('max_fraud_probability', 0) * 100,
                fraud_probability_mean=aggregated_metrics.get('fraud_probability_distribution', {}).get('mean', 0) * 100,
                fraud_probability_std=aggregated_metrics.get('fraud_probability_distribution', {}).get('std', 0) * 100,
                dl_verification_score=dl_score * 100,
                rto_verification_score=rto_score * 100,
                fir_verification_score=fir_score * 100,
                verification_reliability=combined_reliability * 100,
                detailed_analysis=fraud_detection_result,
                annotated_images=fraud_detection_result.get('annotated_images', [])
            )

            # === Save image files ===
            if image_files:
                ClaimDatabaseHandler._save_claim_images(claim, image_files, multi_image_analysis)

            # === Add history logs ===
            ClaimHistory.objects.create(
                claim=claim,
                action='submitted',
                new_status=status,
                performed_by=policyholder.username,
                notes=f"Claim submitted with {claim.total_images_submitted} image(s). Fraud detection: {confidence_score:.1f}% confidence, {risk_level} risk."
            )

            ClaimHistory.objects.create(
                claim=claim,
                action='analyzed',
                performed_by='AI System',
                notes=f"AI Analysis completed. Fraud: {'Yes' if fraud_detected else 'No'}, Confidence: {confidence_score:.1f}%, Risk: {risk_level}"
            )

            # ✅ Return created claim (this must be indented inside function)
            return claim

        except Exception as e:
            print(f"❌ Error creating claim: {e}")
            import traceback
            traceback.print_exc()
            raise

    
    @staticmethod
    def _save_claim_images(claim, image_files, multi_image_analysis):
        """Save individual claim images with their analysis"""
        try:
            individual_images = multi_image_analysis.get('individual_images', [])
            annotated_images = claim.detailed_analysis.get('annotated_images', [])
        
            image_paths = []
        
            for idx, image_file in enumerate(image_files):
            # Get analysis for this specific image
                image_analysis = individual_images[idx] if idx < len(individual_images) else {}
                annotated_data = annotated_images[idx] if idx < len(annotated_images) else {}
            
                damage_analysis = image_analysis.get('damage_analysis', {})
            
            # Create ClaimImage record
                claim_image = ClaimImage.objects.create(
                    claim=claim,
                    image_index=idx + 1,
                    fraud_probability=image_analysis.get('image_fraud_probability', 0) * 100,
                    confidence=image_analysis.get('image_confidence', 0) * 100,
                    damage_percentage=damage_analysis.get('damage_percentage', 0),
                    severity_level=damage_analysis.get('severity_level', 'LOW'),
                    damage_areas_count=image_analysis.get('detection_results', {}).get('high_confidence_detections', 0),
                    damage_regions=damage_analysis.get('damage_regions', [])
                )
            
            # Save original image
                claim_image.image_file.save(
                    f"claim_{claim.claim_number}_img_{idx+1}.jpg",
                    image_file,
                    save=True
                )
            
            # Store the RELATIVE path (URL path), not the absolute file system path
            # This is the key fix - use claim_image.image_file.name instead of .path
                image_paths.append(claim_image.image_file.name)
            
            # Save annotated image if available
                if annotated_data.get('annotated_image_base64'):
                    try:
                        annotated_base64 = annotated_data['annotated_image_base64']
                        annotated_bytes = base64.b64decode(annotated_base64)
                        annotated_file = ContentFile(annotated_bytes)
                    
                        claim_image.annotated_image.save(
                            f"claim_{claim.claim_number}_annotated_{idx+1}.jpg",
                            annotated_file,
                            save=True
                        )
                    except Exception as e:
                        print(f"⚠️ Could not save annotated image {idx+1}: {e}")
        
        # Update claim with image paths (relative paths for URLs)
            claim.image_paths = image_paths
            claim.save(update_fields=['image_paths'])
        
            print(f"✅ Saved {len(image_paths)} images for claim {claim.claim_number}")
            print(f"   Image paths: {image_paths}")
        
        except Exception as e:
            print(f"❌ Error saving claim images: {e}")
            raise
    
    @staticmethod
    def get_claim_by_id(claim_id):
        """Get a specific claim by ID"""
        try:
            return Claim.objects.select_related('policyholder').prefetch_related(
                'images', 'history'
            ).get(id=claim_id)
        except Claim.DoesNotExist:
            return None
    
    @staticmethod
    def get_claim_by_number(claim_number):
        """Get a specific claim by claim number"""
        try:
            return Claim.objects.select_related('policyholder').prefetch_related(
                'images', 'history'
            ).get(claim_number=claim_number)
        except Claim.DoesNotExist:
            return None
    
    @staticmethod
    def get_all_claims(status_filter=None, order_by='-submitted_at'):
        """
        Get all claims, optionally filtered by status
        
        Args:
            status_filter: 'Pending', 'Verified', 'Fraud', 'Rejected' or None for all
            order_by: Field to order by (default: newest first)
        """
        claims = Claim.objects.select_related('policyholder').prefetch_related('images')
        
        if status_filter:
            claims = claims.filter(status=status_filter)
        
        return claims.order_by(order_by)
    
    @staticmethod
    def get_claims_by_policyholder(policyholder_username, status_filter=None):
        """Get all claims for a specific policyholder"""
        claims = Claim.objects.filter(
            policyholder__username=policyholder_username
        ).select_related('policyholder').prefetch_related('images')
        
        if status_filter:
            claims = claims.filter(status=status_filter)
        
        return claims.order_by('-submitted_at')
    
    @staticmethod
    def get_high_risk_claims():
        """Get all high-risk claims requiring review"""
        return Claim.objects.filter(
            risk_level__in=['HIGH', 'MEDIUM']
        ).select_related('policyholder').order_by('-confidence_score')
    
    @staticmethod
    def get_fraud_detected_claims():
        """Get all claims where fraud was detected"""
        return Claim.objects.filter(
            fraud_detected=True
        ).select_related('policyholder').order_by('-submitted_at')
    
    @staticmethod
    @transaction.atomic
    def update_claim_status(claim_id, new_status, admin_user, notes=None):
        """
        Update claim status with history tracking
        
        Args:
            claim_id: Claim ID
            new_status: New status ('Pending', 'Verified', 'Fraud', 'Rejected')
            admin_user: Username of admin making the change
            notes: Optional notes about the status change
        """
        try:
            claim = Claim.objects.get(id=claim_id)
            old_status = claim.status
            
            if old_status == new_status:
                return claim, False  # No change
            
            # Update status
            claim.status = new_status
            claim.save(update_fields=['status', 'updated_at'])
            
            # Determine action type
            if new_status == 'Verified':
                action = 'approved'
            elif new_status == 'Rejected' or new_status == 'Fraud':
                action = 'rejected'
            else:
                action = 'status_changed'
            
            # Create history entry
            ClaimHistory.objects.create(
                claim=claim,
                action=action,
                old_status=old_status,
                new_status=new_status,
                performed_by=admin_user,
                notes=notes or f"Status changed from {old_status} to {new_status}"
            )
            
            return claim, True
            
        except Claim.DoesNotExist:
            return None, False
    
    @staticmethod
    @transaction.atomic
    def add_admin_review(claim_id, admin_user, notes, status_change=None):
        """
        Add admin review notes to a claim
        
        Args:
            claim_id: Claim ID
            admin_user: Username of admin
            notes: Review notes
            status_change: Optional new status to set
        """
        try:
            claim = Claim.objects.get(id=claim_id)
            
            # Update claim with review info
            claim.admin_notes = notes
            claim.reviewed_by = admin_user
            claim.reviewed_at = timezone.now()
            
            if status_change:
                claim.status = status_change
            
            claim.save()
            
            # Create history entry
            ClaimHistory.objects.create(
                claim=claim,
                action='reviewed',
                old_status=claim.status,
                new_status=status_change if status_change else claim.status,
                performed_by=admin_user,
                notes=notes
            )
            
            return claim
            
        except Claim.DoesNotExist:
            return None
    
    @staticmethod
    def get_claim_history(claim_id):
        """Get complete history for a claim"""
        try:
            return ClaimHistory.objects.filter(
                claim_id=claim_id
            ).order_by('-timestamp')
        except Exception as e:
            print(f"Error fetching claim history: {e}")
            return []
    
    @staticmethod
    def get_claim_statistics():
        """Get overall claim statistics"""
        from django.db.models import Count, Avg, Q
        
        stats = Claim.objects.aggregate(
            total_claims=Count('id'),
            pending_claims=Count('id', filter=Q(status='Pending')),
            verified_claims=Count('id', filter=Q(status='Verified')),
            fraud_claims=Count('id', filter=Q(status='Fraud')),
            rejected_claims=Count('id', filter=Q(status='Rejected')),
            
            fraud_detected_count=Count('id', filter=Q(fraud_detected=True)),
            high_risk_count=Count('id', filter=Q(risk_level='HIGH')),
            medium_risk_count=Count('id', filter=Q(risk_level='MEDIUM')),
            low_risk_count=Count('id', filter=Q(risk_level='LOW')),
            
            avg_confidence_score=Avg('confidence_score'),
            avg_claim_amount=Avg('claim_amount'),
        )
        
        return stats
    
    @staticmethod
    def get_policyholder_claim_history(policyholder_username):
        """Get claim history for a specific policyholder"""
        try:
            policyholder = Policyholder.objects.get(username=policyholder_username)
            claims = Claim.objects.filter(policyholder=policyholder).order_by('-submitted_at')
            
            return {
                'policyholder': policyholder,
                'total_claims': claims.count(),
                'pending_claims': claims.filter(status='Pending').count(),
                'verified_claims': claims.filter(status='Verified').count(),
                'fraud_claims': claims.filter(status='Fraud').count(),
                'rejected_claims': claims.filter(status='Rejected').count(),
                'claims': claims
            }
        except Policyholder.DoesNotExist:
            return None
    
    @staticmethod
    def serialize_claim(claim, include_images=True, include_history=True):
        """
        Serialize claim to dictionary for API response
        """
        data = {
            'id': claim.id,
            'claim_number': claim.claim_number,
            'policyholder': {
                'username': claim.policyholder.username,
                'email': claim.policyholder.email,
                'name': f"{claim.policyholder.username}"
            },
        
        # Basic info
            'claim_description': claim.claim_description,
            'accident_date': claim.accident_date.isoformat() if claim.accident_date else None,
            'claim_amount': float(claim.claim_amount),
        
        # Document info
            'dl_number': claim.dl_number,
            'vehicle_reg_no': claim.vehicle_reg_no,
            'fir_number': claim.fir_number,
        
        # Verification scores
            'dl_verification_score': float(claim.dl_verification_score) if claim.dl_verification_score else 0,
            'rto_verification_score': float(claim.rto_verification_score) if claim.rto_verification_score else 0,
            'fir_verification_score': float(claim.fir_verification_score) if claim.fir_verification_score else 0,
        
        # Status
            'status': claim.status,
            'fraud_detected': claim.fraud_detected,
            'confidence_score': float(claim.confidence_score),
            'risk_level': claim.risk_level,
        
        # Scores
            'tabular_fraud_probability': float(claim.tabular_fraud_probability) if claim.tabular_fraud_probability else 0,
            'image_fraud_probability': float(claim.image_fraud_probability) if claim.image_fraud_probability else 0,
            'fusion_score': float(claim.fusion_score) if claim.fusion_score else 0,
        
        # Damage info
            'overall_damage_severity': claim.overall_damage_severity or 'LOW',
            'total_damage_areas': claim.total_damage_areas or 0,
            'average_damage_percentage': float(claim.average_damage_percentage) if claim.average_damage_percentage else 0,
            'max_damage_percentage': float(claim.max_fraud_probability) if claim.max_fraud_probability else 0,
        
        # Images
            'total_images_submitted': claim.total_images_submitted or 0,
        
        # Timestamps
            'submitted_at': claim.submitted_at.isoformat(),
            'updated_at': claim.updated_at.isoformat(),
            'reviewed_at': claim.reviewed_at.isoformat() if claim.reviewed_at else None,
        
        # Admin info
            'reviewed_by': claim.reviewed_by,
            'admin_notes': claim.admin_notes,
        
        # Complete analysis
            'detailed_analysis': claim.detailed_analysis,
        }
    
    # Include individual images if requested
        if include_images:
            images_data = []
            for img in claim.images.all():
                img_data = {
                    'image_index': img.image_index,
                    'fraud_probability': float(img.fraud_probability) if img.fraud_probability else 0,
                    'confidence': float(img.confidence) if img.confidence else 0,
                    'damage_percentage': float(img.damage_percentage) if img.damage_percentage else 0,
                    'severity_level': img.severity_level or 'LOW',
                    'damage_areas_count': img.damage_areas_count or 0,
                    'damage_areas': [region.get('region_id', f'Area {i+1}') if isinstance(region, dict) else str(region) 
                            for i, region in enumerate(img.damage_regions or [])],
                    'image_url': img.image_file.url if img.image_file else None,
                    'annotated_image_url': img.annotated_image.url if img.annotated_image else None,
                }
            
            # Debug: Print image URLs
                if img.image_file:
                    print(f"Image {img.image_index} URL: {img.image_file.url}")
                if img.annotated_image:
                    print(f"Annotated Image {img.image_index} URL: {img.annotated_image.url}")
            
                images_data.append(img_data)
        
            data['images'] = images_data
    
    # Include history if requested
        if include_history:
            data['history'] = [
                {
                    'action': hist.action,
                    'old_status': hist.old_status,
                    'new_status': hist.new_status,
                    'performed_by': hist.performed_by,
                    'notes': hist.notes,
                    'timestamp': hist.timestamp.isoformat()
                }
                for hist in claim.history.all().order_by('-timestamp')
            ]
    
        return data