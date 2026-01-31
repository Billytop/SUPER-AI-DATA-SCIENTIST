import { useState, useEffect } from 'react';
import { useAuthStore } from '../lib/authStore';
import { authAPI, systemAPI } from '../lib/api';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
    User,
    Settings as SettingsIcon,
    Users,
    Bot,
    Database,
    ChevronLeft,
    LogOut,
    Save,
    Globe,
    Cpu,
    Command,
    Shield,
    CreditCard
} from 'lucide-react';

export default function Settings() {
    const { user, fetchUser, logout } = useAuthStore();
    const navigate = useNavigate();
    const [activeSection, setActiveSection] = useState('general');
    const [preferences, setPreferences] = useState({
        theme: 'dark',
        language: 'en',
        notifications: true,
        ai_temperature: 0.7,
        system_prompt: ''
    });

    const [teamMembers, setTeamMembers] = useState<any[]>([]);
    const [dbConfig, setDbConfig] = useState({
        db_name: '',
        db_host: '127.0.0.1',
        db_port: '3306',
        db_user: 'root',
        db_password: ''
    });

    const [message, setMessage] = useState('');
    const [isSaving, setIsSaving] = useState(false);

    useEffect(() => {
        if (user?.profile?.preferences) {
            setPreferences(prev => ({
                ...prev,
                ...user.profile.preferences
            }));
        }
    }, [user]);

    useEffect(() => {
        if (activeSection === 'team') {
            fetchTeam();
        }
    }, [activeSection]);

    const fetchTeam = async () => {
        try {
            const res = await authAPI.getOrganizationUsers();
            setTeamMembers(res.data);
        } catch (error) {
            console.error('Failed to fetch team', error);
        }
    };

    const handleSavePreferences = async () => {
        setIsSaving(true);
        try {
            await authAPI.updatePreferences({ preferences });
            setMessage('Preferences synchronized.');
            fetchUser();
            setTimeout(() => setMessage(''), 3000);
        } catch (error) {
            setMessage('Synchronization failed.');
        } finally {
            setIsSaving(false);
        }
    };

    const handleSaveDbConfig = async () => {
        try {
            await systemAPI.configureDatabase(dbConfig);
            setMessage('Database updated. Restarting node...');
            setTimeout(() => window.location.reload(), 5000);
        } catch (error) {
            setMessage('Configuration rejection.');
        }
    };

    const navItems = [
        { id: 'general', label: 'Identity', icon: User },
        { id: 'security', label: 'Security', icon: Shield },
        { id: 'team', label: 'Network', icon: Users },
        { id: 'ai', label: 'Neural Engine', icon: Bot },
        { id: 'system', label: 'Database', icon: Database },
        { id: 'billing', label: 'Subscription', icon: CreditCard },
    ];

    return (
        <div className="flex h-screen bg-[#030712] text-white overflow-hidden noise-bg">
            {/* Sidebar */}
            <div className="w-72 bg-black/40 border-r border-white/5 flex flex-col backdrop-blur-3xl">
                <div className="p-8 border-b border-white/5 flex items-center gap-4 cursor-pointer group" onClick={() => navigate('/chat')}>
                    <div className="p-2 bg-white/5 rounded-xl group-hover:bg-white/10 transition-all border border-white/5">
                        <ChevronLeft size={20} className="text-white/60" />
                    </div>
                    <div>
                        <h1 className="text-xl font-bold tracking-tighter">System <span className="text-white/40">Core</span></h1>
                        <p className="text-[10px] uppercase tracking-[0.2em] text-white/20 font-black">Preferences</p>
                    </div>
                </div>

                <nav className="flex-1 p-6 space-y-2">
                    {navItems.map((item) => (
                        <button
                            key={item.id}
                            onClick={() => setActiveSection(item.id)}
                            className={`w-full flex items-center space-x-4 px-5 py-4 rounded-2xl text-sm font-bold transition-all relative group ${activeSection === item.id
                                ? 'bg-white/5 text-white border border-white/10 shadow-lg'
                                : 'text-white/40 hover:text-white/70 hover:bg-white/[0.02]'
                                }`}
                        >
                            <item.icon size={18} className={activeSection === item.id ? 'text-white' : 'text-white/20'} />
                            <span className="tracking-tight">{item.label}</span>
                            {activeSection === item.id && (
                                <motion.div layoutId="active" className="absolute left-0 w-1 h-6 bg-white rounded-r-full" />
                            )}
                        </button>
                    ))}
                </nav>

                <div className="p-6 border-t border-white/5">
                    <button
                        onClick={() => { logout(); navigate('/login'); }}
                        className="w-full flex items-center space-x-4 px-5 py-4 rounded-2xl text-sm font-bold text-red-400/60 hover:text-red-400 hover:bg-red-400/5 transition-all border border-transparent hover:border-red-400/10"
                    >
                        <LogOut size={18} />
                        <span className="tracking-tight">Deactivate Session</span>
                    </button>
                </div>
            </div>

            {/* Main Content */}
            <div className="flex-1 overflow-y-auto bg-gradient-to-b from-white/[0.02] to-transparent">
                <div className="max-w-5xl mx-auto p-16">
                    <motion.div
                        key={activeSection}
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.4, ease: "easeOut" }}
                    >
                        <div className="mb-12 flex justify-between items-end">
                            <div>
                                <h2 className="text-5xl font-bold tracking-tighter mb-3 capitalize">
                                    {navItems.find(i => i.id === activeSection)?.label}
                                </h2>
                                <p className="text-white/30 font-medium italic">Configure your enterprise neural parameters.</p>
                            </div>
                            <div className="text-[10px] text-white/10 uppercase tracking-[0.4em] font-black pb-2">
                                Node: {user?.profile?.organization_name || 'UNRESTRICTED'}
                            </div>
                        </div>

                        {message && (
                            <motion.div
                                initial={{ opacity: 0, scale: 0.95 }}
                                animate={{ opacity: 1, scale: 1 }}
                                className={`p-5 mb-8 rounded-2xl border backdrop-blur-xl animate-fadeIn ${message.includes('failed') ? 'bg-red-500/10 border-red-500/20 text-red-400' : 'bg-white/5 border-white/10 text-white/80'}`}
                            >
                                <div className="flex items-center gap-3 font-bold text-sm">
                                    <Shield size={16} />
                                    {message}
                                </div>
                            </motion.div>
                        )}

                        <div className="space-y-8">

                            {/* IDENTITY SECTION */}
                            {activeSection === 'general' && (
                                <div className="glass-card rounded-[32px] p-10 space-y-10">
                                    <div className="flex items-center gap-10">
                                        <div className="w-24 h-24 rounded-3xl bg-white text-black flex items-center justify-center text-4xl font-black shadow-[0_0_50px_rgba(255,255,255,0.1)]">
                                            {user?.first_name?.[0] || 'U'}
                                        </div>
                                        <div className="space-y-1">
                                            <h3 className="text-3xl font-bold tracking-tighter">{user?.first_name} {user?.last_name}</h3>
                                            <p className="text-white/30 font-medium">{user?.email}</p>
                                            <div className="pt-2">
                                                <span className="px-3 py-1 bg-white/5 border border-white/10 rounded-full text-[10px] font-black uppercase tracking-widest text-white/50">Level 4 Access</span>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-2 gap-8 pt-6 border-t border-white/5">
                                        <div className="space-y-2">
                                            <label className="text-[10px] font-bold text-white/20 uppercase tracking-[0.2em] ml-2">Personal Handle</label>
                                            <div className="bg-white/[0.02] border border-white/5 p-4 rounded-2xl text-white/60 font-medium">
                                                @{user?.first_name?.toLowerCase()}
                                            </div>
                                        </div>
                                        <div className="space-y-2">
                                            <label className="text-[10px] font-bold text-white/20 uppercase tracking-[0.2em] ml-2">Assigned Organization</label>
                                            <div className="bg-white/[0.02] border border-white/5 p-4 rounded-2xl text-white/60 font-medium">
                                                {user?.profile?.organization_name || 'Autonomous Operator'}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            )}

                            {/* SECURITY SECTION */}
                            {activeSection === 'security' && (
                                <div className="space-y-6">
                                    <div className="glass-card rounded-[32px] p-10">
                                        <h3 className="text-xl font-bold mb-8 flex items-center gap-3">
                                            <Database size={20} className="text-white/40" />
                                            Authentication Protocols
                                        </h3>
                                        <div className="space-y-4">
                                            <button className="flex items-center justify-between w-full p-6 rounded-2xl border border-white/5 bg-white/[0.02] hover:bg-white/[0.05] transition-all group">
                                                <div className="flex flex-col items-start gap-1">
                                                    <span className="font-bold text-white">Neural Key Modification</span>
                                                    <span className="text-[10px] text-white/20 font-black uppercase tracking-widest">Last rotated: 92 days ago</span>
                                                </div>
                                                <div className="px-4 py-2 bg-white text-black text-[10px] font-black uppercase tracking-widest rounded-full opacity-60 group-hover:opacity-100 transition-all">Rotote Key</div>
                                            </button>

                                            <div className="flex items-center justify-between p-6 rounded-2xl border border-white/5 bg-white/[0.02]">
                                                <div className="flex flex-col gap-1">
                                                    <span className="font-bold text-white">Biometric Proxy (2FA)</span>
                                                    <span className="text-xs text-white/30 font-medium">Secondary identification layer for core overrides.</span>
                                                </div>
                                                <div className="w-12 h-6 bg-white/10 rounded-full relative p-1 cursor-pointer">
                                                    <div className="w-4 h-4 bg-white/20 rounded-full shadow-lg" />
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="glass-card rounded-[32px] p-10">
                                        <h3 className="text-xl font-bold mb-8 flex items-center gap-3">
                                            <Globe size={20} className="text-white/40" />
                                            Active Neural Sessions
                                        </h3>
                                        <div className="space-y-4">
                                            <div className="p-6 rounded-2xl border border-white/5 bg-white/[0.02] flex items-center justify-between">
                                                <div className="flex items-center gap-5">
                                                    <div className="p-3 bg-white/5 rounded-2xl border border-white/10">
                                                        <Cpu size={20} className="text-white/40" />
                                                    </div>
                                                    <div>
                                                        <p className="font-bold text-white">Workstation 01 - Nairobi Node</p>
                                                        <p className="text-[10px] text-white/20 font-black uppercase tracking-[0.2em] mt-1">IP: 197.248.XX.XX • CURRENT SESSION</p>
                                                    </div>
                                                </div>
                                                <span className="px-3 py-1 bg-white/10 border border-white/10 rounded-full text-[9px] font-black uppercase tracking-widest text-white/60">Active Now</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            )}

                            {/* NEURAL ENGINE (AI) */}
                            {activeSection === 'ai' && (
                                <div className="glass-card rounded-[32px] p-10 space-y-12">
                                    <div className="space-y-6">
                                        <div className="flex justify-between items-end">
                                            <label className="text-[10px] font-bold text-white/30 uppercase tracking-[0.2em] ml-2">Inference Entropy (Creativity)</label>
                                            <span className="text-2xl font-bold tracking-tighter text-white">{preferences.ai_temperature}</span>
                                        </div>
                                        <div className="relative pt-2">
                                            <input
                                                type="range"
                                                min="0" max="1" step="0.1"
                                                value={preferences.ai_temperature || 0.7}
                                                onChange={(e) => setPreferences({ ...preferences, ai_temperature: parseFloat(e.target.value) })}
                                                className="w-full h-1 bg-white/5 rounded-full appearance-none cursor-pointer accent-white"
                                            />
                                            <div className="flex justify-between text-[9px] font-black text-white/10 mt-3 uppercase tracking-widest">
                                                <span>Linear (Precise)</span>
                                                <span>Abstract (Creative)</span>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="space-y-4">
                                        <label className="text-[10px] font-bold text-white/30 uppercase tracking-[0.2em] ml-2">Core System Protocol</label>
                                        <textarea
                                            value={preferences.system_prompt || ''}
                                            onChange={(e) => setPreferences({ ...preferences, system_prompt: e.target.value })}
                                            placeholder="Specify neural behavioral parameters..."
                                            className="w-full p-6 bg-white/[0.03] border border-white/10 rounded-3xl h-48 text-white placeholder:text-white/10 focus:outline-none focus:ring-1 focus:ring-white/40 transition-all font-medium leading-relaxed shadow-inner"
                                        />
                                        <div className="flex items-center gap-2 p-4 bg-white/[0.02] border border-white/5 rounded-2xl">
                                            <Shield className="w-4 h-4 text-white/20" />
                                            <p className="text-[10px] text-white/30 font-bold uppercase tracking-wider">Instructions propagate across all inference nodes in this organization.</p>
                                        </div>
                                    </div>

                                    <button
                                        onClick={handleSavePreferences}
                                        className="w-full bg-white text-black font-black py-5 rounded-[24px] hover:bg-gray-200 transition-all flex items-center justify-center gap-3 text-sm uppercase tracking-widest"
                                    >
                                        <Save size={18} />
                                        Initialize Synch
                                    </button>
                                </div>
                            )}

                            {/* DATABASE SECTION */}
                            {activeSection === 'system' && (
                                <div className="glass-card rounded-[32px] p-10 space-y-10 border-indigo-500/20 shadow-[0_30px_60px_-15px_rgba(79,70,229,0.15)]">
                                    <div className="flex items-center gap-4 p-5 bg-indigo-500/5 border border-indigo-500/10 rounded-2xl">
                                        <Database size={24} className="text-indigo-400" />
                                        <div>
                                            <h4 className="font-bold text-indigo-200 tracking-tight">Database Connectivity</h4>
                                            <p className="text-xs text-indigo-400/60 font-medium">Re-routing database channels will trigger a node restart.</p>
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-2 gap-6">
                                        {['db_name', 'db_host', 'db_port', 'db_user'].map((field) => (
                                            <div key={field} className="space-y-2">
                                                <label className="text-[10px] font-bold text-white/20 uppercase tracking-[0.2em] ml-2">{field.replace('_', ' ')}</label>
                                                <input
                                                    type="text"
                                                    value={dbConfig[field as keyof typeof dbConfig]}
                                                    onChange={(e) => setDbConfig({ ...dbConfig, [field]: e.target.value })}
                                                    className="w-full bg-white/[0.03] border border-white/10 rounded-2xl px-6 py-4 text-white font-medium focus:outline-none focus:ring-1 focus:ring-white/40 shadow-inner"
                                                />
                                            </div>
                                        ))}
                                        <div className="space-y-2 col-span-2">
                                            <label className="text-[10px] font-bold text-white/20 uppercase tracking-[0.2em] ml-2">Access Key (Password)</label>
                                            <input
                                                type="password"
                                                value={dbConfig.db_password}
                                                onChange={(e) => setDbConfig({ ...dbConfig, db_password: e.target.value })}
                                                className="w-full bg-white/[0.03] border border-white/10 rounded-2xl px-6 py-4 text-white font-medium focus:outline-none focus:ring-1 focus:ring-white/40 shadow-inner"
                                            />
                                        </div>
                                    </div>

                                    <button
                                        onClick={handleSaveDbConfig}
                                        className="w-full border border-indigo-500/30 bg-indigo-500/10 text-indigo-200 font-black py-5 rounded-[24px] hover:bg-indigo-500/20 transition-all uppercase tracking-widest text-sm"
                                    >
                                        Execute Channel Migration
                                    </button>
                                </div>
                            )}

                        </div>
                    </motion.div>
                </div>
            </div>

            {/* Global Status Bar Overlay */}
            <div className="fixed bottom-8 flex gap-4 text-[10px] text-white/10 uppercase tracking-[0.4em] font-black right-12 z-20">
                <span className="flex items-center gap-1.5"><Globe className="w-3 h-3" />Mesh Secure</span>
                <span className="flex items-center gap-1.5"><Cpu className="w-3 h-3" />Accel Level 4</span>
                <span className="flex items-center gap-1.5"><Command className="w-3 h-3" />v2.5.0-Ent</span>
            </div>
        </div>
    );
}
