import { ArrowUpRight, ArrowDownRight, TrendingUp } from 'lucide-react';
import { motion } from 'framer-motion';
import { cn } from '../lib/utils';

interface KPIProps {
    title: string;
    value: string;
    trend: number; // percentage
    icon: React.ElementType;
    color: 'blue' | 'purple' | 'green' | 'orange';
}

const colorMap = {
    blue: 'from-blue-500 to-cyan-500',
    purple: 'from-purple-500 to-pink-500',
    green: 'from-emerald-500 to-teal-500',
    orange: 'from-orange-500 to-red-500',
};

const bgMap = {
    blue: 'bg-blue-500/10 text-blue-400',
    purple: 'bg-purple-500/10 text-purple-400',
    green: 'bg-emerald-500/10 text-emerald-400',
    orange: 'bg-orange-500/10 text-orange-400',
};

export function KPIWidget({ title, value, trend, icon: Icon, color }: KPIProps) {
    const isPositive = trend >= 0;

    return (
        <motion.div
            whileHover={{ y: -5 }}
            className="bg-gray-900/50 backdrop-blur-xl border border-white/5 p-6 rounded-2xl relative overflow-hidden group"
        >
            <div className={cn("absolute -right-6 -top-6 w-32 h-32 bg-gradient-to-br opacity-5 rounded-full blur-2xl transition-opacity group-hover:opacity-10", colorMap[color])} />

            <div className="flex justify-between items-start mb-4">
                <div className={cn("p-3 rounded-xl", bgMap[color])}>
                    <Icon className="w-6 h-6" />
                </div>
                <div className={cn("flex items-center gap-1 text-sm font-medium px-2 py-1 rounded-full border border-white/5", isPositive ? "text-green-400 bg-green-500/10" : "text-red-400 bg-red-500/10")}>
                    {isPositive ? <ArrowUpRight className="w-3 h-3" /> : <ArrowDownRight className="w-3 h-3" />}
                    {Math.abs(trend)}%
                </div>
            </div>

            <h3 className="text-gray-400 text-sm font-medium mb-1">{title}</h3>
            <div className="text-2xl font-bold text-white tracking-tight">{value}</div>
        </motion.div>
    );
}
