import { Home, BarChart2, MessageSquare, Settings, Users, Box, ShoppingCart, Activity } from 'lucide-react';
import { Link, useLocation } from 'react-router-dom';
import { cn } from '../lib/utils';
import { motion } from 'framer-motion';

const menuItems = [
    { icon: Home, label: 'Dashboard', path: '/' },
    { icon: ShoppingCart, label: 'Sales', path: '/sales' },
    { icon: Box, label: 'Inventory', path: '/inventory' },
    { icon: Users, label: 'Partners', path: '/partners' },
    { icon: Activity, label: 'Operations', path: '/operations' },
    { icon: BarChart2, label: 'Analytics', path: '/analytics' },
    { icon: Settings, label: 'Settings', path: '/settings' },
];

export function Sidebar() {
    const location = useLocation();

    return (
        <motion.div
            initial={{ x: -20, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            className="h-screen w-64 glass-panel border-r-0 text-white p-4 fixed left-0 top-0 flex flex-col z-50 rounded-r-2xl"
        >
            <div className="flex items-center gap-2 mb-8 px-2">
                <div className="h-8 w-8 bg-gradient-to-tr from-blue-500 to-purple-500 rounded-lg" />
                <span className="text-xl font-bold tracking-tight">Sephlighty<span className="text-blue-400">AI</span></span>
            </div>

            <nav className="space-y-2 flex-1">
                {menuItems.map((item) => {
                    const isActive = location.pathname === item.path;
                    return (
                        <Link
                            key={item.path}
                            to={item.path}
                            className={cn(
                                "flex items-center gap-3 px-3 py-2 rounded-xl transition-all duration-200 group relative overflow-hidden",
                                isActive
                                    ? "bg-white/10 text-white shadow-lg shadow-blue-500/10"
                                    : "text-gray-400 hover:text-white hover:bg-white/5"
                            )}
                        >
                            {isActive && (
                                <motion.div
                                    layoutId="active-pill"
                                    className="absolute inset-0 bg-gradient-to-r from-blue-600/20 to-purple-600/20"
                                />
                            )}
                            <item.icon className={cn("w-5 h-5 relative z-10", isActive ? "text-blue-400" : "group-hover:text-blue-300")} />
                            <span className="relative z-10 font-medium">{item.label}</span>
                        </Link>
                    );
                })}
            </nav>

            <div className="p-4 bg-gradient-to-br from-blue-900/20 to-purple-900/20 rounded-2xl border border-white/5">
                <div className="flex items-center gap-3">
                    <div className="h-10 w-10 rounded-full bg-gray-700 flex items-center justify-center">
                        <Users className="w-5 h-5 text-gray-300" />
                    </div>
                    <div>
                        <p className="text-sm font-medium">Admin User</p>
                        <p className="text-xs text-gray-400">admin@sephlighty.ai</p>
                    </div>
                </div>
            </div>
        </motion.div>
    );
}
