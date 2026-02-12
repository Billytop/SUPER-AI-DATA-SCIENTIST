import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuthStore } from '../lib/authStore';
import { motion } from 'framer-motion';
import { Brain, Mail, Lock, Building, ArrowRight, Shield, Globe, Cpu, Command, Orbit, Database, Activity } from 'lucide-react';

export default function Register() {
    const [formData, setFormData] = useState({
        organization_name: '',
        email: '',
        password: '',
        first_name: '',
        last_name: '',
    });
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    const { register } = useAuthStore();
    const navigate = useNavigate();

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError('');
        setLoading(true);

        try {
            await register(formData);
            navigate('/chat');
        } catch (err: any) {
            const data = err.response?.data;
            if (data) {
                const messages = Object.keys(data).map(key => {
                    const val = data[key];
                    return Array.isArray(val) ? `${key}: ${val[0]}` : val;
                }).join('. ');
                setError(messages || 'Handshake failed. Protocol mismatch.');
            } else {
                setError('Connection failure. Neural node registration timed out.');
            }
        } finally {
            setLoading(false);
        }
    };

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    return (
        <div className="h-screen w-full bg-[#02040a] relative flex items-center justify-center overflow-hidden noise-bg font-sans">
            <div className="scanline"></div>

            {/* Immersive Neural Background */}
            <div className="absolute inset-0 pointer-events-none data-grid opacity-20"></div>

            <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="absolute inset-0 pointer-events-none"
            >
                <div className="absolute top-[-10%] right-[-10%] w-[50%] h-[50%] bg-white/4 rounded-full blur-[160px] animate-slow-pan"></div>
                <div className="absolute bottom-[-10%] left-[-10%] w-[50%] h-[50%] bg-indigo-500/5 rounded-full blur-[160px] animate-slow-pan fill-mode-reverse"></div>
            </motion.div>

            <div className="w-full max-w-[1240px] px-8 relative z-20 py-12">
                <motion.div
                    initial={{ opacity: 0, scale: 0.98 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.8 }}
                    className="grid lg:grid-cols-[1fr_1.2fr] glass-panel bg-black/40 rounded-[40px] overflow-hidden shadow-2xl border border-white/5"
                >

                    {/* Left Panel: Initialization Highlights */}
                    <div className="hidden lg:flex flex-col p-16 border-r border-white/5 relative bg-gradient-to-b from-white/[0.02] to-transparent">
                        <motion.div
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: 0.2 }}
                            className="flex items-center gap-4 mb-20"
                        >
                            <div className="w-14 h-14 bg-white rounded-[20px] flex items-center justify-center shadow-glow">
                                <Brain className="w-8 h-8 text-black" />
                            </div>
                            <div className="space-y-0.5">
                                <span className="text-2xl font-black tracking-tighter text-white uppercase">Sephlighty</span>
                                <p className="text-[10px] font-bold text-white/30 uppercase tracking-[0.3em]">Neural Core v4.0</p>
                            </div>
                        </motion.div>

                        <div className="space-y-12">
                            <motion.div
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: 0.3 }}
                                className="space-y-4"
                            >
                                <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 text-[10px] font-black tracking-[0.2em] text-white/40 uppercase">
                                    Node-Entry Protocol
                                </div>
                                <h2 className="text-6xl font-black tracking-tighter text-white leading-[0.9]">
                                    Integrate <br />
                                    Your <br />
                                    <span className="text-white/20 italic">Logic.</span>
                                </h2>
                            </motion.div>

                            <div className="space-y-6">
                                {[
                                    { title: 'Military Encryption', desc: 'RSA-4096 handshake protocol standard.', icon: Shield },
                                    { title: 'Mesh Scaling', desc: 'Instant propagation across 672 nodes.', icon: Activity },
                                    { title: 'Neural Audit', desc: 'Immutable traceability for every inference.', icon: Database }
                                ].map((item, i) => (
                                    <motion.div
                                        initial={{ opacity: 0, x: -20 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        transition={{ delay: 0.4 + (i * 0.1) }}
                                        key={i}
                                        className="flex gap-5 p-6 rounded-3xl bg-white/[0.02] border border-white/5 hover:bg-white/[0.04] transition-all group cursor-pointer"
                                    >
                                        <item.icon className="w-5 h-5 text-white/20 shrink-0 mt-1 group-hover:text-white transition-all" />
                                        <div>
                                            <div className="font-black text-white text-[11px] uppercase tracking-widest">{item.title}</div>
                                            <div className="text-sm text-white/30 mt-1 font-medium">{item.desc}</div>
                                        </div>
                                    </motion.div>
                                ))}
                            </div>
                        </div>

                        <div className="mt-auto pt-16 flex items-center gap-4 text-white/10 text-[10px] font-black uppercase tracking-[0.5em]">
                            <Orbit className="w-5 h-5 animate-spin-slow" /> Protocol Deployment Ready
                        </div>
                    </div>

                    {/* Right Panel: Entity Synchronization */}
                    <div className="p-12 lg:p-20 flex flex-col justify-center bg-[#02040a]/60">
                        <div className="max-w-md mx-auto w-full space-y-10">
                            <motion.div
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                transition={{ delay: 0.5 }}
                                className="space-y-2"
                            >
                                <h3 className="text-4xl font-black text-white tracking-tighter uppercase leading-none">Entity Initializer</h3>
                                <p className="text-white/30 font-bold italic tracking-tight">Create your unique organizational neural node.</p>
                            </motion.div>

                            {error && (
                                <motion.div
                                    initial={{ opacity: 0, x: -10 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    className="p-5 bg-red-500/10 border border-red-500/20 rounded-2xl text-red-400 text-sm font-bold flex items-center gap-4"
                                >
                                    <Shield className="w-5 h-5 shrink-0" />
                                    {error}
                                </motion.div>
                            )}

                            <form onSubmit={handleSubmit} className="space-y-6">
                                <div className="space-y-2">
                                    <label className="text-[11px] font-black text-white/20 uppercase tracking-[0.3em] ml-2">Organizational ID</label>
                                    <div className="relative group">
                                        <Building className="absolute left-6 top-1/2 -translate-y-1/2 w-5 h-5 text-white/10 group-focus-within:text-indigo-400 transition-colors" />
                                        <input
                                            type="text"
                                            name="organization_name"
                                            value={formData.organization_name}
                                            onChange={handleChange}
                                            className="w-full bg-white/[0.02] border border-white/5 rounded-3xl pl-16 pr-6 py-5 text-white placeholder:text-white/5 focus:outline-none focus:ring-1 focus:ring-white/20 focus:bg-white/[0.04] transition-all font-bold"
                                            placeholder="Entity Name"
                                            required
                                        />
                                    </div>
                                </div>

                                <div className="grid grid-cols-2 gap-4">
                                    <div className="space-y-2">
                                        <label className="text-[11px] font-black text-white/20 uppercase tracking-[0.3em] ml-2">Given Name</label>
                                        <input
                                            type="text"
                                            name="first_name"
                                            value={formData.first_name}
                                            onChange={handleChange}
                                            className="w-full bg-white/[0.02] border border-white/5 rounded-3xl px-6 py-5 text-white placeholder:text-white/5 focus:outline-none focus:ring-1 focus:ring-white/20 transition-all font-bold"
                                            placeholder="Operator"
                                            required
                                        />
                                    </div>
                                    <div className="space-y-2">
                                        <label className="text-[11px] font-black text-white/20 uppercase tracking-[0.3em] ml-2">Surname</label>
                                        <input
                                            type="text"
                                            name="last_name"
                                            value={formData.last_name}
                                            onChange={handleChange}
                                            className="w-full bg-white/[0.02] border border-white/5 rounded-3xl px-6 py-5 text-white placeholder:text-white/5 focus:outline-none focus:ring-1 focus:ring-white/20 transition-all font-bold"
                                            placeholder="ID"
                                            required
                                        />
                                    </div>
                                </div>

                                <div className="space-y-2">
                                    <label className="text-[11px] font-black text-white/20 uppercase tracking-[0.3em] ml-2">Communication Link</label>
                                    <div className="relative group">
                                        <Mail className="absolute left-6 top-1/2 -translate-y-1/2 w-5 h-5 text-white/10 group-focus-within:text-indigo-400 transition-colors" />
                                        <input
                                            type="email"
                                            name="email"
                                            value={formData.email}
                                            onChange={handleChange}
                                            className="w-full bg-white/[0.02] border border-white/5 rounded-3xl pl-16 pr-6 py-5 text-white placeholder:text-white/5 focus:outline-none focus:ring-1 focus:ring-white/20 focus:bg-white/[0.04] transition-all font-bold"
                                            placeholder="primary@neural.mesh"
                                            required
                                        />
                                    </div>
                                </div>

                                <div className="space-y-2">
                                    <label className="text-[11px] font-black text-white/20 uppercase tracking-[0.3em] ml-2">Access Sequence Key</label>
                                    <div className="relative group">
                                        <Lock className="absolute left-6 top-1/2 -translate-y-1/2 w-5 h-5 text-white/10 group-focus-within:text-indigo-400 transition-colors" />
                                        <input
                                            type="password"
                                            name="password"
                                            value={formData.password}
                                            onChange={handleChange}
                                            className="w-full bg-white/[0.02] border border-white/5 rounded-3xl pl-16 pr-6 py-5 text-white placeholder:text-white/5 focus:outline-none focus:ring-1 focus:ring-white/20 focus:bg-white/[0.04] transition-all font-bold"
                                            placeholder="Entropy req: 8+ chars"
                                            required
                                            minLength={8}
                                        />
                                    </div>
                                </div>

                                <button
                                    type="submit"
                                    disabled={loading}
                                    className="w-full mt-8 bg-white text-black font-black py-7 rounded-[28px] hover:bg-gray-100 active:scale-[0.96] transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-glow flex items-center justify-center gap-6 text-xl group"
                                >
                                    {loading ? (
                                        <div className="w-8 h-8 border-4 border-black/20 border-t-black rounded-full animate-spin"></div>
                                    ) : (
                                        <>
                                            INITIALIZE NODE
                                            <ArrowRight className="w-6 h-6 transform group-hover:translate-x-1.5 transition-transform" />
                                        </>
                                    )}
                                </button>
                            </form>

                            <div className="mt-12 text-center">
                                <p className="text-white/20 text-[11px] font-black uppercase tracking-[0.3em]">
                                    Active node already registered?{' '}
                                    <Link to="/login" className="text-white hover:text-indigo-400 transition-all border-b border-white/20 pb-0.5 ml-2 hover:border-indigo-400/50">
                                        Sign In
                                    </Link>
                                </p>
                            </div>
                        </div>
                    </div>
                </motion.div>

                {/* Global Footer Security Stats */}
                <div className="mt-12 flex flex-wrap justify-center gap-12 text-[10px] text-white/5 uppercase tracking-[0.5em] font-black">
                    <span className="flex items-center gap-3"><Orbit className="w-4 h-4" /> Global Mesh Active</span>
                    <span className="flex items-center gap-3"><Cpu className="w-4 h-4" /> Hardware Accel Level 4</span>
                    <span className="flex items-center gap-3"><Globe className="w-4 h-4" /> East African Region</span>
                    <span className="flex items-center gap-3"><Command className="w-4 h-4" /> Kernel v4.2.0-Ent</span>
                </div>
            </div>
        </div>
    );
}
