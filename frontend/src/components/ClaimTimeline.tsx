"use client";

import React, { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { CheckCircle, AlertCircle, ChevronDown, ChevronUp } from "lucide-react";

interface TimelineStep {
  step: string;
  status: string;
  timestamp: string | null;
  performed_by: string;
  notes: string;
  is_completed: boolean;
}

interface ClaimTimelineData {
  claim_id: number;
  claim_number: string;
  current_status: string;
  submitted_at: string;
  timeline_steps: TimelineStep[];
}

interface ClaimTimelineProps {
  claimId: number;
  onClose?: () => void;
}

// Status badge variants
const statusBadgeVariants: Record<string, "default" | "secondary" | "destructive" | "outline"> = {
  Submitted: "default",
  "AI Analysis": "secondary",
  "Under Review": "secondary",
  "Field Survey": "secondary",
  Decision: "destructive",
  Pending: "outline",
};

const formatDate = (dateString: string | null | undefined): string => {
  if (!dateString) return "Pending";
  try {
    const date = new Date(dateString);
    return date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return "Invalid date";
  }
};

export const ClaimTimeline: React.FC<ClaimTimelineProps> = ({
  claimId,
  onClose,
}) => {
  const [timeline, setTimeline] = useState<ClaimTimelineData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedStep, setExpandedStep] = useState<number | null>(0);

  useEffect(() => {
    fetchTimeline();
  }, [claimId]);

  const fetchTimeline = async () => {
    try {
      setLoading(true);
      setError(null);

      const token = localStorage.getItem("access_token");
      if (!token) {
        setError("Authentication token not found");
        return;
      }

      const response = await fetch(
        `http://127.0.0.1:8000/api/detection/claims/${claimId}/timeline/`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
        }
      );

      if (!response.ok) {
        if (response.status === 403) {
          setError("You don't have permission to view this claim's timeline");
        } else if (response.status === 404) {
          setError("Claim not found");
        } else {
          setError("Failed to fetch claim timeline");
        }
        return;
      }

      const data = await response.json();
      
      // If claim is in a final state, mark all steps as completed
      const finalStatuses = ['Verified', 'Rejected', 'Fraud', 'Approved', 'Survey Completed'];
      if (finalStatuses.includes(data.current_status)) {
        data.timeline_steps = data.timeline_steps.map((step: TimelineStep) => ({
          ...step,
          is_completed: true,
        }));
      }
      
      setTimeline(data);
    } catch (err) {
      setError("An error occurred while fetching the timeline");
      console.error("Timeline fetch error:", err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center py-12">
          <div className="space-y-3 w-full">
            <div className="h-4 bg-gray-200 rounded animate-pulse"></div>
            <div className="h-4 bg-gray-200 rounded animate-pulse w-5/6"></div>
            <div className="h-4 bg-gray-200 rounded animate-pulse w-4/6"></div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }

  if (!timeline) {
    return null;
  }

  const completedCount = timeline.timeline_steps.filter(
    (step) => step.is_completed
  ).length;
  const totalSteps = timeline.timeline_steps.length;
  const progressPercent = (completedCount / totalSteps) * 100;

  return (
    <div className="space-y-6">
      {/* Header Card */}
      <Card>
        <CardHeader className="pb-6">
          <div className="space-y-4">
            <div>
              <CardTitle className="text-2xl mb-2">Claim Timeline</CardTitle>
              <p className="text-sm text-muted-foreground">
                Claim #{timeline.claim_number}
              </p>
            </div>

            {/* Progress Info */}
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium">Progress</span>
                <span className="text-sm text-muted-foreground">
                  {completedCount} of {totalSteps} steps completed
                </span>
              </div>
              <div className="w-full bg-secondary rounded-full h-2.5 overflow-hidden">
                <div
                  className="bg-green-600 h-2.5 rounded-full transition-all duration-500"
                  style={{ width: `${progressPercent}%` }}
                />
              </div>
            </div>

            {/* Current Status */}
            <div className="flex items-center gap-3">
              <span className="text-sm font-medium">Current Status:</span>
              <Badge variant="outline">{timeline.current_status}</Badge>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Timeline Steps */}
      <div className="space-y-3">
        {timeline.timeline_steps.map((step, index) => (
          <div key={index}>
            <Collapsible
              open={expandedStep === index}
              onOpenChange={(open) => setExpandedStep(open ? index : null)}
            >
              <CollapsibleTrigger asChild>
                <Button
                  variant="outline"
                  className="w-full justify-between h-auto py-4 px-4 rounded-lg hover:bg-accent"
                >
                  <div className="flex items-center gap-4 flex-1 text-left">
                    {step.is_completed ? (
                      <CheckCircle className="h-5 w-5 text-green-600 flex-shrink-0" />
                    ) : (
                      <div className="h-5 w-5 rounded-full border-2 border-gray-300 flex-shrink-0" />
                    )}
                    <div className="flex-1 min-w-0">
                      <p className="font-semibold text-sm">{step.step}</p>
                      <p className="text-xs text-muted-foreground">
                        {step.is_completed
                          ? formatDate(step.timestamp)
                          : "Pending"}
                      </p>
                    </div>
                  </div>
                  {expandedStep === index ? (
                    <ChevronUp className="h-4 w-4 flex-shrink-0" />
                  ) : (
                    <ChevronDown className="h-4 w-4 flex-shrink-0" />
                  )}
                </Button>
              </CollapsibleTrigger>

              <CollapsibleContent className="pt-0">
                <Card className="border-t-0 rounded-t-none mt-0">
                  <CardContent className="pt-6 pb-6 space-y-4">
                    {/* Status */}
                    <div>
                      <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
                        Status
                      </p>
                      <Badge variant="secondary">{step.status}</Badge>
                    </div>

                    <Separator />

                    {/* Performed By */}
                    {step.is_completed && step.performed_by && (
                      <div>
                        <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
                          Performed By
                        </p>
                        <p className="text-sm text-foreground">{step.performed_by}</p>
                      </div>
                    )}

                    {/* Notes */}
                    {step.notes && step.notes !== "Pending" && (
                      <>
                        {step.performed_by && step.is_completed && <Separator />}
                        <div>
                          <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
                            Details
                          </p>
                          <p className="text-sm text-foreground">
                            {step.notes}
                          </p>
                        </div>
                      </>
                    )}

                    {/* Timestamp */}
                    {step.is_completed && step.timestamp && (
                      <>
                        <Separator />
                        <div>
                          <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
                            Date & Time
                          </p>
                          <p className="text-sm text-foreground">
                            {formatDate(step.timestamp)}
                          </p>
                        </div>
                      </>
                    )}
                  </CardContent>
                </Card>
              </CollapsibleContent>
            </Collapsible>
          </div>
        ))}
      </div>

      {/* Final Status Message */}
      {completedCount === totalSteps ? (
        <Alert>
          <CheckCircle className="h-4 w-4" />
          <AlertDescription>
            All steps completed! Your claim has been fully processed.
          </AlertDescription>
        </Alert>
      ) : (
        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            Your claim is currently in progress. {totalSteps - completedCount}{" "}
            step{totalSteps - completedCount !== 1 ? "s" : ""} remaining.
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
};

export default ClaimTimeline;
