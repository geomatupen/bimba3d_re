import { X, LogIn } from "lucide-react";
import { useAuth } from "../../context/AuthContext";

export default function LoginModal() {
  const { isLoginOpen, closeModals, loginWithGoogle } = useAuth();
  if (!isLoginOpen) return null;
  return (
    <div className="fixed inset-0 z-50">
      <div className="absolute inset-0 bg-black/60" onClick={closeModals} />
      <div className="absolute inset-0 flex items-center justify-center p-4">
        <div className="w-full max-w-md bg-white rounded-xl shadow-2xl border border-slate-200 overflow-hidden">
          <div className="flex items-center justify-between px-4 py-3 border-b border-slate-200">
            <h3 className="text-base font-semibold text-slate-900">Login</h3>
            <button className="p-2 rounded-lg hover:bg-slate-100" onClick={closeModals}>
              <X className="w-5 h-5 text-slate-700" />
            </button>
          </div>
          <div className="px-4 py-6 space-y-3">
            <button
              className="w-full inline-flex items-center justify-center gap-2 px-4 py-2 rounded-lg border border-slate-300 hover:bg-slate-50"
              onClick={loginWithGoogle}
            >
              <LogIn className="w-4 h-4" />
              Continue with Google (stub)
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
