import { Sidebar } from '../components/Sidebar';
import { AIChat } from '../components/AIChat';
import { KPIWidget } from '../components/KPIWidget';
import { SalesChart } from '../components/SalesChart';
import { DollarSign, ShoppingBag, Users, Activity } from 'lucide-react';
import { motion } from 'framer-motion';

export default function Dashboard() {
    return (
        <div className="min-h-screen bg-black text-white font-sans selection:bg-blue-500/30">
            <Sidebar />

            <main className="ml-64 p-8">
                <header className="mb-10 flex justify-between items-center">
                    <div>
                        <h1 className="text-3xl font-bold bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
                            Welcome back, Admin
                        </h1>
                        <p className="text-gray-400 mt-1">Here's what's happening with your business today.</p>
                    </div>

                    <div className="flex gap-4">
                        {/* Date Picker or other actions could go here */}
                        <div className="px-4 py-2 bg-white/5 rounded-lg border border-white/10 text-sm text-gray-300">
                            {new Date().toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}
                        </div>
                    </div>
                </header>

                {/* KPI Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                    <KPIWidget title="Total Revenue" value="$45,231.89" trend={12.5} icon={DollarSign} color="blue" />
                    <KPIWidget title="Total Orders" value="1,205" trend={8.2} icon={ShoppingBag} color="purple" />
                    <KPIWidget title="Active Customers" value="892" trend={-2.4} icon={Users} color="orange" />
                    <KPIWidget title="Stock Level" value="14,034" trend={5.1} icon={Activity} color="green" />
                </div>

                {/* Charts Section Placeholder */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.1 }}
                        className="lg:col-span-2 bg-gray-900/50 backdrop-blur-xl border border-white/5 rounded-2xl p-6 h-[400px]"
                    >
                        <h3 className="text-lg font-semibold mb-6">Revenue Analytics</h3>
                        <div className="h-full -mt-6">
                            <SalesChart />
                        </div>
                    </motion.div>

                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.2 }}
                        className="bg-gray-900/50 backdrop-blur-xl border border-white/5 rounded-2xl p-6 h-[400px]"
                    >
                        <h3 className="text-lg font-semibold mb-6">Recent Activity</h3>
                        <div className="space-y-4">
                            {[1, 2, 3, 4, 5].map(i => (
                                <div key={i} className="flex items-center gap-4 p-3 hover:bg-white/5 rounded-lg transition-colors cursor-pointer">
                                    <div className="w-10 h-10 rounded-full bg-blue-500/10 flex items-center justify-center text-blue-400">
                                        <ShoppingBag className="w-4 h-4" />
                                    </div>
                                    <div>
                                        <p className="text-sm font-medium">New Order #102{i}</p>
                                        <p className="text-xs text-gray-500">2 minutes ago</p>
                                    </div>
                                    <div className="ml-auto font-mono text-sm">+$124.00</div>
                                </div>
                            ))}
                        </div>
                    </motion.div>
                </div>
            </main>

            <AIChat />
        </div>
    );
}
