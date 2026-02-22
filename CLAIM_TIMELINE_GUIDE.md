# Claim Timeline View - Implementation Guide

## Overview

The Claim Timeline View is a feature that allows customers to see a visual step-by-step progress tracker for their insurance claims. The timeline shows the entire claim journey from submission through final decision.

## Features

- **Visual Timeline Display**: Shows 5 main steps:
  1. **Submitted** - Claim submitted by customer
  2. **AI Analysis** - Automated AI analysis completed
  3. **Under Review** - Admin review in progress
  4. **Field Survey** - Surveyor conducting field assessment
  5. **Decision** - Final decision made

- **Timeline Information**:
  - Step name and status
  - Date and time of completion
  - Who performed the action (staff member, system)
  - Additional notes or details
  - Completion status (completed/pending)

- **Progress Tracking**:
  - Visual progress bar showing completion percentage
  - Current status badge
  - Indicator for pending vs completed steps

## Backend Implementation

### 1. Database Models

**Existing Models Used:**
- `Claim` - Main claim record with status fields
- `ClaimHistory` - Tracks all actions/events on a claim

**Key Fields in ClaimHistory:**
```python
- action: 'submitted', 'analyzed', 'reviewed', 'surveyor_assigned', 'survey_completed', etc.
- timestamp: When the action occurred
- performed_by: Who performed the action
- old_status / new_status: Status changes
- notes: Additional details
```

### 2. API Endpoint

**Endpoint:** `GET /api/detection/claims/<claim_id>/timeline/`

**Authentication:** Required (JWT Token)

**Permissions:** User must be the policyholder or admin

**Response Format:**
```json
{
  "claim_id": 1,
  "claim_number": "CLM-20250222120000-5",
  "current_status": "Under Review",
  "submitted_at": "2025-02-22T12:00:00Z",
  "timeline_steps": [
    {
      "step": "Submitted",
      "status": "Pending",
      "timestamp": "2025-02-22T12:00:00Z",
      "performed_by": "Customer",
      "notes": "",
      "is_completed": true
    },
    {
      "step": "AI Analysis",
      "status": "Pending",
      "timestamp": "2025-02-22T12:05:00Z",
      "performed_by": "System",
      "notes": "Low fraud risk detected",
      "is_completed": true
    },
    {
      "step": "Under Review",
      "status": "Pending",
      "timestamp": null,
      "performed_by": "",
      "notes": "Pending",
      "is_completed": false
    }
  ]
}
```

### 3. Serializers

New serializers added in `serializers.py`:
- `ClaimTimelineStepSerializer` - Represents a single timeline step
- `ClaimTimelineSerializer` - Complete timeline data

### 4. Creating Timeline History Records

When events occur, create `ClaimHistory` records:

```python
from .models import ClaimHistory

# When claim is submitted
ClaimHistory.objects.create(
    claim=claim,
    action='submitted',
    performed_by=request.user.username,
    notes='Claim submitted by customer'
)

# When AI analysis completes
ClaimHistory.objects.create(
    claim=claim,
    action='analyzed',
    performed_by='System',
    new_status='Pending',
    notes=f'AI Analysis: Risk Level {claim.risk_level}'
)

# When admin reviews
ClaimHistory.objects.create(
    claim=claim,
    action='reviewed',
    old_status=old_status,
    new_status=new_status,
    performed_by=admin_user.username,
    notes=admin_notes
)

# When surveyor is assigned
ClaimHistory.objects.create(
    claim=claim,
    action='surveyor_assigned',
    performed_by=admin_user.username,
    notes=f'Assigned to {surveyor.username}'
)

# When survey completes
ClaimHistory.objects.create(
    claim=claim,
    action='survey_completed',
    performed_by=surveyor.username,
    new_status='Survey Completed',
    notes='Field survey completed'
)
```

## Frontend Implementation

### 1. Timeline Component

**File:** `frontend/src/components/ClaimTimeline.tsx`

**Features:**
- Fetches timeline data from backend API
- Displays visual timeline with icons and steps
- Shows progress bar with percentage
- Responsive design (works on mobile and desktop)
- Error handling and loading states

**Props:**
```typescript
interface ClaimTimelineProps {
  claimId: number;           // ID of the claim to display
  onClose?: () => void;      // Optional callback when closing
}
```

**Usage:**
```tsx
import ClaimTimeline from "@/components/ClaimTimeline";

// In a modal
<ClaimTimeline claimId={claimId} onClose={handleClose} />

// On a page
<ClaimTimeline claimId={claimId} />
```

### 2. Integration Points

#### In Customer Dashboard (`frontend/src/app/customer/page.tsx`)

1. **Timeline Button in Claims Table**
   - Desktop: Added to actions column with Timeline icon
   - Mobile: Added to actions button group

2. **Timeline Modal**
   - Modal overlay that displays when "Timeline" button is clicked
   - Can be closed by clicking X or outside the modal

3. **Dedicated Timeline Page**
   - Route: `/claim-timeline/[claimId]`
   - Full-screen view of the timeline
   - Back button to return to claims list

### 3. Visual Styling

**Icons per Step:**
- Submitted: FileCheck icon
- AI Analysis: BarChart3 icon
- Under Review: AlertCircle icon
- Field Survey: MapPin icon
- Decision: CheckCircle icon

**Colors:**
- Completed steps: Green background with checkmark
- Pending steps: Gray background with clock icon
- Progress bar: Blue

**Status Badges:**
- Different colors for different claim statuses
- Shows current status at top of timeline

## Usage Flow

### For Customers

1. **View All Claims**: Navigate to customer dashboard
2. **Click Timeline Button**: On any claim in the list
3. **View Timeline**: See the claim's progress journey
4. **Close Timeline**: Click X or outside modal, or navigate back

### For Developers

1. **Ensure ClaimHistory Records are Created**
   - Every significant action should create a history record
   - Use the action constants: 'submitted', 'analyzed', 'reviewed', etc.

2. **Update Claim Status Appropriately**
   - When status changes, update the `Claim.status` field
   - Create corresponding `ClaimHistory` record with old_status and new_status

3. **Include Performer Information**
   - Record who performed each action (username, system, etc.)
   - This helps track accountability

4. **Add Context in Notes**
   - Include relevant details in `notes` field
   - Examples: "Low fraud risk", "Assigned to surveyor John", etc.

## API Endpoint Details

### Route Registration

**File:** `backend/detection/urls.py`

```python
path('claims/<int:claim_id>/timeline/', get_claim_timeline, name='get-claim-timeline'),
```

### Error Handling

The endpoint returns appropriate HTTP status codes:
- `200 OK` - Timeline data retrieved successfully
- `403 Forbidden` - User doesn't have permission to view this claim
- `404 Not Found` - Claim not found
- `500 Internal Server Error` - Server error

### Authentication

Requires valid JWT token in Authorization header:
```
Authorization: Bearer <access_token>
```

## Testing

### Backend Testing

1. Create a test claim
2. Create multiple ClaimHistory records with different actions
3. Call the endpoint: `GET /api/detection/claims/<claim_id>/timeline/`
4. Verify the response includes all steps in correct order

### Frontend Testing

1. Log in as a customer
2. Navigate to claims dashboard
3. Click "Timeline" button on a claim
4. Verify timeline displays correctly
5. Test modal open/close functionality
6. Test on mobile and desktop

## Future Enhancements

1. **Timeline Filters**: Filter by action type or date range
2. **Export Timeline**: Download timeline as PDF
3. **Email Notifications**: Send email when timeline step completes
4. **Timeline History**: Show full history of actions (not just milestones)
5. **Timeline Comments**: Allow customers to add comments at each step
6. **Custom Steps**: Configure different claim types with different steps
7. **Estimated Completion**: Show estimated dates for pending steps

## Troubleshooting

### Timeline Not Loading

**Issue**: Timeline component shows "Loading..." indefinitely

**Solution**:
1. Check that JWT token is stored in `localStorage.access_token`
2. Verify the backend API is running
3. Check browser console for errors
4. Ensure claim ID is valid

### History Records Not Appearing

**Issue**: ClaimHistory records not being created

**Solution**:
1. Check that `ClaimHistory.objects.create()` is being called
2. Verify the claim object exists before creating history
3. Check database migrations have run
4. Verify `performed_by` field is populated

### Permission Denied Error

**Issue**: "You don't have permission to view this claim's timeline"

**Solution**:
1. Ensure logged-in user is the claim's policyholder
2. Verify claim belongs to the correct policyholder
3. Admin/superuser should always have access

## Files Modified/Created

### Backend
- `backend/detection/serializers.py` - Added ClaimTimelineSerializer, ClaimTimelineStepSerializer
- `backend/detection/claim_views.py` - Added get_claim_timeline endpoint
- `backend/detection/urls.py` - Added URL route for timeline endpoint

### Frontend
- `frontend/src/components/ClaimTimeline.tsx` - NEW: Timeline component
- `frontend/src/app/customer/page.tsx` - Modified: Added timeline button and modal
- `frontend/src/app/claim-timeline/[claimId]/page.tsx` - NEW: Dedicated timeline page
