"use client";

import { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
    Send, MessageCircle, Calendar, Clock, MapPin, User, CheckCircle, XCircle,
    AlertCircle
} from "lucide-react";

const API_BASE = `/api/detection`;

interface Message {
    id: number;
    sender: string;
    sender_role: string;
    recipient: string;
    message: string;
    timestamp: string;
    is_read: boolean;
    message_type: string;
}

interface Appointment {
    id: number;
    claim_id: number;
    claim_number: string;
    customer: string;
    surveyor: string;
    proposed_datetime: string;
    duration_minutes: number;
    status: string;
    location: string;
    notes: string;
    created_at: string;
    confirmed_at?: string;
    completed_at?: string;
}

interface Claim {
    id: number;
    claim_number: string;
    status: string;
    assigned_surveyor_name?: string;
}

interface CustomerChatAppointmentProps {
    claim: Claim;
}

export default function CustomerChatAppointment({ claim }: CustomerChatAppointmentProps) {
    const [messages, setMessages] = useState<Message[]>([]);
    const [newMessage, setNewMessage] = useState("");
    const [loading, setLoading] = useState(false);
    const [appointments, setAppointments] = useState<Appointment[]>([]);
    const [showAppointmentDialog, setShowAppointmentDialog] = useState(false);
    const [appointmentForm, setAppointmentForm] = useState({
        proposed_datetime: "",
        location: "",
        notes: ""
    });
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    useEffect(() => {
        if (claim) {
            fetchMessages();
            fetchAppointments();
        }
    }, [claim]);

    const fetchMessages = async () => {
        try {
            const token = localStorage.getItem("access_token");
            if (!token) return;

            const res = await fetch(`${API_BASE}/chat/${claim.id}/messages/`, {
                headers: { "Authorization": `Bearer ${token}` }
            });

            if (res.ok) {
                const data = await res.json();
                setMessages(data.messages || []);
            }
        } catch (err) {
            console.error("Failed to fetch messages:", err);
        }
    };

    const fetchAppointments = async () => {
        try {
            const token = localStorage.getItem("access_token");
            if (!token) return;

            const res = await fetch(`${API_BASE}/appointments/`, {
                headers: { "Authorization": `Bearer ${token}` }
            });

            if (res.ok) {
                const data = await res.json();
                const claimAppointments = data.appointments?.filter((apt: Appointment) => apt.claim_id === claim.id) || [];
                setAppointments(claimAppointments);
            }
        } catch (err) {
            console.error("Failed to fetch appointments:", err);
        }
    };

    const sendMessage = async () => {
        if (!newMessage.trim()) return;

        try {
            const token = localStorage.getItem("access_token");
            if (!token) return;

            const res = await fetch(`${API_BASE}/chat/${claim.id}/send/`, {
                method: "POST",
                headers: {
                    "Authorization": `Bearer ${token}`,
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ message: newMessage.trim() })
            });

            if (res.ok) {
                setNewMessage("");
                fetchMessages();
            }
        } catch (err) {
            console.error("Failed to send message:", err);
        }
    };

    const createAppointment = async () => {
        if (!appointmentForm.proposed_datetime || !appointmentForm.location) {
            alert("Please fill in all required fields");
            return;
        }

        try {
            const token = localStorage.getItem("access_token");
            if (!token) return;

            const res = await fetch(`${API_BASE}/appointments/create/`, {
                method: "POST",
                headers: {
                    "Authorization": `Bearer ${token}`,
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    claim_id: claim.id,
                    ...appointmentForm
                })
            });

            if (res.ok) {
                setShowAppointmentDialog(false);
                setAppointmentForm({ proposed_datetime: "", location: "", notes: "" });
                fetchAppointments();
                fetchMessages();
            } else {
                const error = await res.json();
                alert(error.error || "Failed to create appointment");
            }
        } catch (err) {
            console.error("Failed to create appointment:", err);
        }
    };

    const confirmAppointment = async (appointmentId: number) => {
        try {
            const token = localStorage.getItem("access_token");
            if (!token) return;

            const res = await fetch(`${API_BASE}/appointments/${appointmentId}/confirm/`, {
                method: "POST",
                headers: { "Authorization": `Bearer ${token}` }
            });

            if (res.ok) {
                fetchAppointments();
                fetchMessages();
            }
        } catch (err) {
            console.error("Failed to confirm appointment:", err);
        }
    };

    const cancelAppointment = async (appointmentId: number) => {
        const reason = prompt("Please provide a reason for cancellation:");
        if (!reason) return;

        try {
            const token = localStorage.getItem("access_token");
            if (!token) return;

            const res = await fetch(`${API_BASE}/appointments/${appointmentId}/cancel/`, {
                method: "POST",
                headers: {
                    "Authorization": `Bearer ${token}`,
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ reason })
            });

            if (res.ok) {
                fetchAppointments();
                fetchMessages();
            }
        } catch (err) {
            console.error("Failed to cancel appointment:", err);
        }
    };

    const formatDateTime = (dateTimeString: string) => {
        return new Date(dateTimeString).toLocaleString();
    };

    const getStatusColor = (status: string) => {
        switch (status) {
            case "scheduled": return "bg-yellow-100 text-yellow-800";
            case "confirmed": return "bg-green-100 text-green-800";
            case "completed": return "bg-blue-100 text-blue-800";
            case "cancelled": return "bg-red-100 text-red-800";
            case "rescheduled": return "bg-purple-100 text-purple-800";
            default: return "bg-gray-100 text-gray-800";
        }
    };

    return (
        <div className="flex flex-col gap-6 h-full w-full">
            {/* Chat Section */}
            <Card className="border border-gray-300 bg-white">
                <CardHeader className="border-b border-gray-200 bg-gray-50">
                    <CardTitle className="flex items-center gap-2 text-gray-900">
                        <MessageCircle className="h-5 w-5" />
                        Chat with Surveyor
                        {claim.assigned_surveyor_name ? (
                            <Badge variant="outline" className="border-gray-400 text-gray-700">{claim.assigned_surveyor_name}</Badge>
                        ) : (
                            <Badge variant="outline" className="border-gray-400 text-gray-600">Not Assigned</Badge>
                        )}
                    </CardTitle>
                </CardHeader>
                <CardContent className="bg-white">
                    {!claim.assigned_surveyor_name ? (
                        <div className="text-center py-8">
                            <AlertCircle className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                            <p className="text-gray-600">No surveyor assigned to this claim yet.</p>
                            <p className="text-sm text-gray-500 mt-2">Chat will be available once a surveyor is assigned.</p>
                        </div>
                    ) : (
                        <>
                            {/* Messages */}
                            <div className="h-96 overflow-y-auto border border-gray-300 rounded p-4 mb-4 bg-gray-50">
                                {messages.length === 0 ? (
                                    <div className="text-center text-gray-500 py-8">
                                        <MessageCircle className="h-8 w-8 mx-auto mb-2 text-gray-300" />
                                        <p>No messages yet. Start the conversation!</p>
                                    </div>
                                ) : (
                                    messages.map((msg) => (
                                        <div
                                            key={msg.id}
                                            className={`mb-4 ${msg.sender_role === 'customer' ? 'text-right' : 'text-left'}`}
                                        >
                                            <div className={`inline-block max-w-xs lg:max-w-md ${msg.sender_role === 'customer' ? 'bg-gray-900 text-white' : 'bg-white border border-gray-300'}`}>
                                                <div className="p-3 rounded">
                                                    <div className="flex items-center gap-2 mb-1">
                                                        <User className="h-3 w-3" />
                                                        <span className="text-xs font-semibold">
                                                            {msg.sender_role === 'customer' ? 'You' : msg.sender}
                                                        </span>
                                                        {msg.message_type === 'appointment' && (
                                                            <Calendar className="h-3 w-3 text-gray-500" />
                                                        )}
                                                    </div>
                                                    <p className="text-sm">{msg.message}</p>
                                                    <p className={`text-xs mt-1 ${msg.sender_role === 'customer' ? 'text-gray-300' : 'text-gray-500'}`}>
                                                        {formatDateTime(msg.timestamp)}
                                                    </p>
                                                </div>
                                            </div>
                                        </div>
                                    ))
                                )}
                                <div ref={messagesEndRef} />
                            </div>

                            {/* Message Input */}
                            <div className="flex gap-2">
                                <Input
                                    value={newMessage}
                                    onChange={(e) => setNewMessage(e.target.value)}
                                    placeholder="Type your message..."
                                    onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                                    disabled={!claim.assigned_surveyor_name}
                                    className="border-gray-300"
                                />
                                <Button 
                                    onClick={sendMessage} 
                                    disabled={!claim.assigned_surveyor_name || !newMessage.trim()}
                                    className="bg-gray-900 text-white hover:bg-gray-800 disabled:bg-gray-400"
                                >
                                    <Send className="h-4 w-4" />
                                </Button>
                            </div>
                        </>
                    )}
                </CardContent>
            </Card>

            {/* Appointments Section */}
            <Card className="border border-gray-300 bg-white">
                <CardHeader className="border-b border-gray-200 bg-gray-50">
                    <CardTitle className="flex items-center gap-2 text-gray-900">
                        <Calendar className="h-5 w-5" />
                        Appointments
                    </CardTitle>
                </CardHeader>
                <CardContent className="bg-white">
                    {!claim.assigned_surveyor_name ? (
                        <div className="text-center py-8">
                            <AlertCircle className="h-8 w-8 text-gray-400 mx-auto mb-2" />
                            <p className="text-gray-600 text-sm">No surveyor assigned yet</p>
                        </div>
                    ) : (
                        <>
                            {/* Create Appointment Button */}
                            <Dialog open={showAppointmentDialog} onOpenChange={setShowAppointmentDialog}>
                                <DialogTrigger asChild>
                                    <Button className="w-full mb-4 bg-gray-900 text-white hover:bg-gray-800 disabled:bg-gray-400" disabled={!claim.assigned_surveyor_name}>
                                        <Calendar className="h-4 w-4 mr-2" />
                                        Schedule Appointment
                                    </Button>
                                </DialogTrigger>
                                <DialogContent>
                                    <DialogHeader>
                                        <DialogTitle className="text-gray-900">Schedule Appointment</DialogTitle>
                                        <DialogDescription className="text-gray-600">
                                            Propose a time for the surveyor to visit and assess the damage.
                                        </DialogDescription>
                                    </DialogHeader>
                                    <div className="space-y-4">
                                        <div>
                                            <Label htmlFor="datetime" className="text-gray-700">Proposed Date & Time</Label>
                                            <Input
                                                id="datetime"
                                                type="datetime-local"
                                                value={appointmentForm.proposed_datetime}
                                                onChange={(e) => setAppointmentForm({...appointmentForm, proposed_datetime: e.target.value})}
                                                min={new Date().toISOString().slice(0, 16)}
                                                className="border-gray-300"
                                            />
                                        </div>
                                        <div>
                                            <Label htmlFor="location" className="text-gray-700">Location</Label>
                                            <Input
                                                id="location"
                                                placeholder="Enter the address where the survey should take place"
                                                value={appointmentForm.location}
                                                onChange={(e) => setAppointmentForm({...appointmentForm, location: e.target.value})}
                                                className="border-gray-300"
                                            />
                                        </div>
                                        <div>
                                            <Label htmlFor="notes" className="text-gray-700">Additional Notes (Optional)</Label>
                                            <Textarea
                                                id="notes"
                                                placeholder="Any special instructions or details for the surveyor"
                                                value={appointmentForm.notes}
                                                onChange={(e) => setAppointmentForm({...appointmentForm, notes: e.target.value})}
                                                className="border-gray-300"
                                            />
                                        </div>
                                        <div className="flex gap-2">
                                            <Button variant="outline" onClick={() => setShowAppointmentDialog(false)} className="border-gray-300 text-gray-700 hover:bg-gray-50">
                                                Cancel
                                            </Button>
                                            <Button onClick={createAppointment} className="bg-gray-900 text-white hover:bg-gray-800">
                                                Send Proposal
                                            </Button>
                                        </div>
                                    </div>
                                </DialogContent>
                            </Dialog>

                            {/* Appointments List */}
                            <div className="space-y-3">
                                {appointments.length === 0 ? (
                                    <div className="text-center py-4">
                                        <Calendar className="h-8 w-8 text-gray-300 mx-auto mb-2" />
                                        <p className="text-sm text-gray-600">No appointments scheduled</p>
                                    </div>
                                ) : (
                                    appointments.map((apt) => (
                                        <div key={apt.id} className="border border-gray-300 rounded p-3 bg-white">
                                            <div className="flex items-center justify-between mb-2">
                                                <Badge className={`${
                                                    apt.status === 'scheduled' ? 'bg-gray-100 text-gray-800 border-gray-300' :
                                                    apt.status === 'confirmed' ? 'bg-gray-100 text-gray-800 border-gray-300' :
                                                    apt.status === 'completed' ? 'bg-gray-100 text-gray-800 border-gray-300' :
                                                    apt.status === 'cancelled' ? 'bg-gray-100 text-gray-800 border-gray-300' :
                                                    'bg-gray-100 text-gray-800 border-gray-300'
                                                }`}>
                                                    {apt.status}
                                                </Badge>
                                                <span className="text-xs text-gray-500">
                                                    {formatDateTime(apt.created_at)}
                                                </span>
                                            </div>
                                            <div className="text-sm text-gray-700">
                                                <p className="font-medium">{formatDateTime(apt.proposed_datetime)}</p>
                                                <p className="text-gray-600">{apt.location}</p>
                                                <p className="text-gray-600">{apt.duration_minutes} minutes</p>
                                                {apt.notes && <p className="text-gray-500 italic">{apt.notes}</p>}
                                            </div>
                                            <div className="flex gap-2 mt-2">
                                                {apt.status === 'scheduled' && (
                                                    <>
                                                        <Button size="sm" onClick={() => confirmAppointment(apt.id)} className="bg-gray-900 text-white hover:bg-gray-800">
                                                            <CheckCircle className="h-3 w-3 mr-1" />
                                                            Confirm
                                                        </Button>
                                                        <Button size="sm" variant="outline" onClick={() => cancelAppointment(apt.id)} className="border-gray-300 text-gray-700 hover:bg-gray-50">
                                                            <XCircle className="h-3 w-3 mr-1" />
                                                            Cancel
                                                        </Button>
                                                    </>
                                                )}
                                                {apt.status !== 'cancelled' && apt.status !== 'completed' && (
                                                    <Button size="sm" variant="outline" onClick={() => cancelAppointment(apt.id)} className="border-gray-300 text-gray-700 hover:bg-gray-50">
                                                        <XCircle className="h-3 w-3 mr-1" />
                                                        Cancel
                                                    </Button>
                                                )}
                                            </div>
                                        </div>
                                    ))
                                )}
                            </div>
                        </>
                    )}
                </CardContent>
            </Card>
        </div>
    );
}
