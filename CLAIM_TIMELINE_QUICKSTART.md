# Claim Timeline View - Quick Start Guide

## What Was Implemented

A complete **Claim Timeline View** feature that allows customers to see a visual step-by-step progress tracker for their insurance claims. The timeline shows 5 main stages:

```
Submitted → AI Analysis → Under Review → Field Survey → Decision
```

Each step displays:
- ✓ Date and time of completion
- ✓ Who performed the action (person or system)
- ✓ Additional notes or context
- ✓ Status information

## Quick Start

### For Customers

1. **Log in** to the customer portal
2. **Go to Claims** tab
3. **Find your claim** in the list
4. **Click "Timeline"** button (blue button with timeline icon)
5. **View the progress** in the modal or dedicated page
6. **See colored steps** showing completed vs pending stages

### For Developers

#### Backend Setup

No additional setup required! The feature uses existing models:
- `Claim` model
- `ClaimHistory` model

#### Frontend Setup

The Timeline component is already integrated into:
- Customer dashboard claims table
- Mobile-friendly card view
- Dedicated timeline page at `/claim-timeline/[claimId]`

## Features

### Visual Timeline Display
```
✓ Progress bar showing completion percentage
✓ Colored status badges
✓ Icons for each step type
✓ Timestamp and performer information
✓ Responsive design (mobile & desktop)
```

### Timeline Steps Tracked
1. **Submitted** - Claim submitted by customer
2. **AI Analysis** - Automated analysis completed  
3. **Under Review** - Under admin review
4. **Field Survey** - Surveyor field assessment
5. **Decision** - Final decision made

## API Endpoint

**URL:** `GET /api/detection/claims/{claim_id}/timeline/`

**Headers:**
```
Authorization: Bearer {access_token}
Content-Type: application/json
```

**Response:**
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
    },
    {
      "step": "Field Survey",
      "status": "Pending",
      "timestamp": null,
      "performed_by": "",
      "notes": "Pending",
      "is_completed": false
    },
    {
      "step": "Decision",
      "status": "Pending",
      "timestamp": null,
      "performed_by": "",
      "notes": "Pending",
      "is_completed": false
    }
  ]
}
```

## Testing the Feature

### Test Scenario 1: View Timeline in Modal
1. Go to customer dashboard
2. Switch to "Claims" tab
3. Click "Timeline" button on any claim
4. Modal should open showing the timeline
5. Click X to close modal

### Test Scenario 2: View Timeline Page
1. Navigate to `/claim-timeline/1` (replace 1 with actual claim ID)
2. Full-screen timeline should display
3. Click "Back to Claims" to return

### Test Scenario 3: Verify Timeline Steps
1. Open timeline for a claim
2. Check that at least "Submitted" and "AI Analysis" are marked as completed
3. Verify timestamps are correct
4. Verify progress bar reflects completed steps

### Test Scenario 4: Mobile View
1. Open customer dashboard on mobile
2. Go to Claims tab
3. Click Timeline button on a claim card
4. Timeline should be responsive and readable

## Files Created/Modified

### New Files
```
frontend/src/components/ClaimTimeline.tsx        - Main timeline component
frontend/src/app/claim-timeline/[claimId]/page.tsx  - Dedicated timeline page
CLAIM_TIMELINE_GUIDE.md                         - Detailed documentation
CLAIM_TIMELINE_QUICKSTART.md                    - This file
```

### Modified Files
```
backend/detection/serializers.py     - Added timeline serializers
backend/detection/claim_views.py     - Added timeline endpoint
backend/detection/urls.py            - Added timeline route
frontend/src/app/customer/page.tsx   - Added timeline buttons and modal
```

## Important: Ensuring History Records are Created

For the timeline to display correctly, **ClaimHistory records must be created** whenever claim events occur.

### Key Events That Should Create History Records

1. **When claim is submitted:**
```python
ClaimHistory.objects.create(
    claim=claim,
    action='submitted',
    performed_by=request.user.username,
    notes='Claim submitted by customer'
)
```

2. **When AI analysis completes:**
```python
ClaimHistory.objects.create(
    claim=claim,
    action='analyzed',
    performed_by='System',
    new_status='Pending',
    notes=f'AI Analysis Complete - Risk: {claim.risk_level}'
)
```

3. **When admin reviews claim:**
```python
ClaimHistory.objects.create(
    claim=claim,
    action='reviewed',
    old_status=old_status,
    new_status=new_status,
    performed_by=admin_user.username,
    notes=admin_notes
)
```

4. **When surveyor is assigned:**
```python
ClaimHistory.objects.create(
    claim=claim,
    action='surveyor_assigned',
    performed_by=admin_user.username,
    notes=f'Assigned to: {surveyor.username}'
)
```

5. **When survey is completed:**
```python
ClaimHistory.objects.create(
    claim=claim,
    action='survey_completed',
    performed_by=surveyor.username,
    new_status='Survey Completed',
    notes='Field survey completed'
)
```

### Checking if Records are Being Created

1. Go to Django admin panel
2. Navigate to "Claim Histories"
3. Filter by Claim
4. You should see multiple records for different actions
5. Each record should have:
   - action (submitted, analyzed, reviewed, etc.)
   - timestamp
   - performed_by
   - notes

## Customizing the Timeline

### Change Step Names

Edit `frontend/src/components/ClaimTimeline.tsx`:
```tsx
const step_flow = [
    ('submitted', 'Claim Filed'),           // Change 'Submitted'
    ('analyzed', 'System Check'),           // Change 'AI Analysis'
    ('reviewed', 'Manual Review'),          // Change 'Under Review'
    ('surveyor_assigned', 'Field Visit'),   // Change 'Field Survey'
    ('survey_completed', 'Final Decision'), // Change 'Decision'
]
```

### Change Colors

Edit step colors in the component:
```tsx
const statusColors: Record<string, string> = {
    Submitted: "text-blue-600",      // Blue
    "AI Analysis": "text-purple-600", // Purple
    // ... etc
}
```

### Change Icons

Edit icons per step:
```tsx
const stepIcons: Record<string, React.ReactNode> = {
    Submitted: <FileCheck className="w-6 h-6" />,
    "AI Analysis": <BarChart3 className="w-6 h-6" />,
    // ... etc
}
```

## Troubleshooting

### Timeline Shows No Steps

**Issue:** Timeline displays but all steps show as pending

**Solution:**
1. Check ClaimHistory records exist in database
2. Verify records have correct `action` values
3. Ensure timestamps are populated
4. Check API response in browser DevTools

### Timeline Won't Load

**Issue:** Loading spinner shows indefinitely

**Solution:**
1. Check browser console for errors
2. Verify JWT token in localStorage
3. Check network tab - ensure API returns 200 OK
4. Verify claim ID is valid

### Steps Out of Order

**Issue:** Timeline steps appear in wrong order

**Solution:**
1. Backend queries ClaimHistory by timestamp
2. Ensure all history records have correct timestamps
3. Check database timezone settings

### Mobile Timeline Cut Off

**Issue:** Timeline too wide on mobile

**Solution:**
- All components use responsive Tailwind classes
- Should auto-adapt to screen size
- Test with mobile browser DevTools

## Performance Considerations

- **Caching**: Timeline data is fetched on demand (no caching)
- **Database**: Uses simple ClaimHistory query (indexed by claim_id)
- **API**: Returns only necessary fields
- **Frontend**: Component renders efficiently

## Security Notes

✓ Only claim owner (policyholder) or admin can view timeline  
✓ User permission verified in backend  
✓ JWT authentication required  
✓ No sensitive data exposed  

## Next Steps

1. **Deploy to production:**
   - Run Django migrations
   - Collect static files
   - Restart backend server

2. **Monitor usage:**
   - Check load times
   - Monitor API response times
   - Collect user feedback

3. **Future enhancements:**
   - Add email notifications per step
   - Export timeline as PDF
   - Add timeline comments
   - Timeline analytics

## Support

For issues or questions:
1. Check the detailed guide: `CLAIM_TIMELINE_GUIDE.md`
2. Review implementation files
3. Check Django logs for errors
4. Check browser console for frontend errors
