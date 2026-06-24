"use client";

import { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
    Send, MessageCircle, Calendar, Clock, MapPin, User, CheckCircle, XCircle,
    AlertCircle, Phone, Mail, Camera, Paperclip, Users, Settings
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
    customer_preferences?: any;
}

interface Claim {
    id: number;
    claim_number: string;
    status: string;
    customer?: string;
}

interface SurveyorChatAppointmentProps {
    claim: Claim;
}

export default function SurveyorChatAppointment({ claim }: SurveyorChatAppointmentProps) {
    const [messages, setMessages] = useState<Message[]>([]);
    const [newMessage, setNewMessage] = useState("");
    const [loading, setLoading] = useState(false);
    const [appointments, setAppointments] = useState<Appointment[]>([]);
    const [showAvailabilityDialog, setShowAvailabilityDialog] = useState(false);
    const [availabilitySlots, setAvailabilitySlots] = useState<Array<{
        date: string;
        start_time: string;
        end_time: string;
        max_appointments: number;
    }>>([]);
    const [newSlot, setNewSlot] = useState({
        date: "",
        start_time: "",
        end_time: "",
        max_appointments: 1
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
            fetchAvailability();
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

    const fetchAvailability = async () => {
        try {
            const token = localStorage.getItem("access_token");
            if (!token) return;

            const res = await fetch(`${API_BASE}/surveyor/availability/`, {
                headers: { "Authorization": `Bearer ${token}` }
            });

            if (res.ok) {
                const data = await res.json();
                setAvailabilitySlots(data.availability_slots || []);
            }
        } catch (err) {
            console.error("Failed to fetch availability:", err);
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

    const updateAppointment = async (appointmentId: number, updates: Partial<Appointment>) => {
        try {
            const token = localStorage.getItem("access_token");
            if (!token) return;

            const res = await fetch(`${API_BASE}/appointments/${appointmentId}/update/`, {
                method: "POST",
                headers: {
                    "Authorization": `Bearer ${token}`,
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(updates)
            });

            if (res.ok) {
                fetchAppointments();
                fetchMessages();
            }
        } catch (err) {
            console.error("Failed to update appointment:", err);
        }
    };

    const setAvailability = async () => {
        try {
            const token = localStorage.getItem("access_token");
            if (!token) return;

            const res = await fetch(`${API_BASE}/surveyor/availability/set/`, {
                method: "POST",
                headers: {
                    "Authorization": `Bearer ${token}`,
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ slots: [newSlot] })
            });

            if (res.ok) {
                setShowAvailabilityDialog(false);
                setNewSlot({ date: "", start_time: "", end_time: "", max_appointments: 1 });
                fetchAvailability();
            }
        } catch (err) {
            console.error("Failed to set availability:", err);
        }
    };

    const addAvailabilitySlot = () => {
        if (!newSlot.date || !newSlot.start_time || !newSlot.end_time) {
            alert("Please fill in all fields");
            return;
        }
        setAvailability();
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
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Chat Section */}
            <Card className="lg:col-span-2">
                <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                        <MessageCircle className="h-5 w-5" />
                        Chat with Customer
                        <Badge variant="outline">{claim.customer || "Customer"}</Badge>
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    {/* Messages */}
                    <div className="h-96 overflow-y-auto border rounded-lg p-4 mb-4 bg-gray-50">
                        {messages.length === 0 ? (
                            <div className="text-center text-gray-500 py-8">
                                <MessageCircle className="h-8 w-8 mx-auto mb-2 text-gray-300" />
                                <p>No messages yet. Start the conversation!</p>
                            </div>
                        ) : (
                            messages.map((msg) => (
                                <div
                                    key={msg.id}
                                    className={`mb-4 ${msg.sender_role === 'surveyor' ? 'text-right' : 'text-left'}`}
                                >
                                    <div className={`inline-block max-w-xs lg:max-w-md ${msg.sender_role === 'surveyor' ? 'bg-blue-500 text-white' : 'bg-white border'}`}>
                                        <div className="p-3 rounded-lg">
                                            <div className="flex items-center gap-2 mb-1">
                                                <User className="h-3 w-3" />
                                                <span className="text-xs font-semibold">
                                                    {msg.sender_role === 'surveyor' ? 'You' : msg.sender}
                                                </span>
                                                {msg.message_type === 'appointment' && (
                                                    <Calendar className="h-3 w-3 text-orange-500" />
                                                )}
                                            </div>
                                            <p className="text-sm">{msg.message}</p>
                                            <p className={`text-xs mt-1 ${msg.sender_role === 'surveyor' ? 'text-blue-100' : 'text-gray-500'}`}>
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
                        />
                        <Button onClick={sendMessage} disabled={!newMessage.trim()}>
                            <Send className="h-4 w-4" />
                        </Button>
                    </div>
                </CardContent>
            </Card>

            {/* Appointments & Availability Section */}
            <div className="space-y-6">
                {/* Appointments */}
                <Card>
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <Calendar className="h-5 w-5" />
                            Appointments
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="space-y-3">
                            {appointments.length === 0 ? (
                                <div className="text-center py-4">
                                    <Calendar className="h-8 w-8 text-gray-300 mx-auto mb-2" />
                                    <p className="text-sm text-gray-600">No appointments scheduled</p>
                                </div>
                            ) : (
                                appointments.map((apt) => (
                                    <div key={apt.id} className="border rounded-lg p-3">
                                        <div className="flex items-center justify-between mb-2">
                                            <Badge className={getStatusColor(apt.status)}>
                                                {apt.status}
                                            </Badge>
                                            <span className="text-xs text-gray-500">
                                                {formatDateTime(apt.created_at)}
                                            </span>
                                        </div>
                                        
                                        <div className="space-y-2 text-sm">
                                            <div className="flex items-center gap-2">
                                                <Clock className="h-3 w-3" />
                                                <span>{formatDateTime(apt.proposed_datetime)}</span>
                                            </div>
                                            <div className="flex items-center gap-2">
                                                <MapPin className="h-3 w-3" />
                                                <span className="text-xs">{apt.location}</span>
                                            </div>
                                            {apt.notes && (
                                                <p className="text-xs text-gray-600 italic">{apt.notes}</p>
                                            )}
                                        </div>

                                        <div className="flex gap-2 mt-3">
                                            {apt.status === 'scheduled' && (
                                                <Button size="sm" onClick={() => confirmAppointment(apt.id)}>
                                                    <CheckCircle className="h-3 w-3 mr-1" />
                                                    Confirm
                                                </Button>
                                            )}
                                            {apt.status !== 'cancelled' && apt.status !== 'completed' && (
                                                <Button size="sm" variant="outline" onClick={() => cancelAppointment(apt.id)}>
                                                    <XCircle className="h-3 w-3 mr-1" />
                                                    Cancel
                                                </Button>
                                            )}
                                        </div>
                                    </div>
                                ))
                            )}
                        </div>
                    </CardContent>
                </Card>

                {/* Availability Management */}
                <Card>
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <Settings className="h-5 w-5" />
                            Your Availability
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        <Dialog open={showAvailabilityDialog} onOpenChange={setShowAvailabilityDialog}>
                            <DialogTrigger asChild>
                                <Button className="w-full mb-4" variant="outline">
                                    <Calendar className="h-4 w-4 mr-2" />
                                    Add Available Time
                                </Button>
                            </DialogTrigger>
                            <DialogContent>
                                <DialogHeader>
                                    <DialogTitle>Set Available Time</DialogTitle>
                                    <DialogDescription>
                                        Add time slots when you're available for appointments.
                                    </DialogDescription>
                                </DialogHeader>
                                <div className="space-y-4">
                                    <div>
                                        <Label htmlFor="date">Date</Label>
                                        <Input
                                            id="date"
                                            type="date"
                                            value={newSlot.date}
                                            onChange={(e) => setNewSlot({...newSlot, date: e.target.value})}
                                            min={new Date().toISOString().split('T')[0]}
                                        />
                                    </div>
                                    <div className="grid grid-cols-2 gap-4">
                                        <div>
                                            <Label htmlFor="start_time">Start Time</Label>
                                            <Input
                                                id="start_time"
                                                type="time"
                                                value={newSlot.start_time}
                                                onChange={(e) => setNewSlot({...newSlot, start_time: e.target.value})}
                                            />
                                        </div>
                                        <div>
                                            <Label htmlFor="end_time">End Time</Label>
                                            <Input
                                                id="end_time"
                                                type="time"
                                                value={newSlot.end_time}
                                                onChange={(e) => setNewSlot({...newSlot, end_time: e.target.value})}
                                            />
                                        </div>
                                    </div>
                                    <div>
                                        <Label htmlFor="max_appointments">Max Appointments</Label>
                                        <Select value={newSlot.max_appointments.toString()} onValueChange={(value) => setNewSlot({...newSlot, max_appointments: parseInt(value)})}>
                                            <SelectTrigger>
                                                <SelectValue />
                                            </SelectTrigger>
                                            <SelectContent>
                                                <SelectItem value="1">1 Appointment</SelectItem>
                                                <SelectItem value="2">2 Appointments</SelectItem>
                                                <SelectItem value="3">3 Appointments</SelectItem>
                                            </SelectContent>
                                        </Select>
                                    </div>
                                    <div className="flex gap-2">
                                        <Button variant="outline" onClick={() => setShowAvailabilityDialog(false)}>
                                            Cancel
                                        </Button>
                                        <Button onClick={addAvailabilitySlot}>
                                            Add Time Slot
                                        </Button>
                                    </div>
                                </div>
                            </DialogContent>
                        </Dialog>

                        {/* Available Slots List */}
                        <div className="space-y-2">
                            {availabilitySlots.length === 0 ? (
                                <p className="text-sm text-gray-600 text-center py-4">No availability slots set</p>
                            ) : (
                                availabilitySlots.slice(0, 5).map((slot, index) => (
                                    <div key={index} className="flex items-center justify-between p-2 border rounded text-sm">
                                        <div>
                                            <p className="font-medium">{slot.date}</p>
                                            <p className="text-gray-600">{slot.start_time} - {slot.end_time}</p>
                                        </div>
                                        <Badge variant="outline">
                                            {slot.max_appointments} max
                                        </Badge>
                                    </div>
                                ))
                            )}
                        </div>
                    </CardContent>
                </Card>
            </div>
        </div>
    );
}
