import { FileText, Sparkles } from "lucide-react";

export default function Navbar() {
  return (
    <header className="sticky top-0 z-50 w-full px-6 py-4">
      <div className="mx-auto flex max-w-[1600px] items-center justify-between glass rounded-2xl px-6 py-3 shadow-lg">
        <div className="flex items-center gap-2.5 group cursor-pointer">
          <div className="bg-primary p-2 rounded-lg text-white shadow-lg shadow-primary/20 transition-transform group-hover:scale-110">
            <FileText size={22} strokeWidth={2.5} />
          </div>
          <h1 className="text-xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-slate-900 to-slate-600">
            PaperMind<span className="text-primary">.ai</span>
          </h1>
        </div>

        <div className="flex items-center gap-4">
            <div className="hidden md:flex items-center gap-1.5 px-3 py-1.5 bg-slate-100 rounded-full text-[13px] font-medium text-slate-600 border border-slate-200">
               <Sparkles size={14} className="text-amber-500" />
               AI Research Assistant
            </div>
        </div>
      </div>
    </header>
  );
}
