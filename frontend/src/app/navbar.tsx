import { FileText } from "lucide-react";

export default function Navbar() {
  return (
    <div className="border-b-slate-200 border py-6 mb-6">
      <div className="container mx-auto flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <FileText />
          <h2 className="text-2xl font-semibold">PaperMind</h2>
        </div>
      </div>
    </div>
  );
}
