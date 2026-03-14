import { useState } from "react";
import { User, LogOut, LogIn } from "lucide-react";
import { useAuth } from "../context/AuthContext";
import LoginModal from "./auth/LoginModal";

export default function UserMenu() {
  const { user, logout, openLogin } = useAuth();
  const [open, setOpen] = useState(false);
  return (
    <div className="relative">
      {user ? (
        <button className="flex items-center gap-2 px-3 py-2 rounded-xl bg-white/10 hover:bg-white/20 border border-white/20 text-white" onClick={() => setOpen((o) => !o)}>
          {user.avatarUrl ? (
            <img src={user.avatarUrl} alt={user.name} className="w-6 h-6 rounded-full border border-white/30" />
          ) : (
            <User className="w-5 h-5" />
          )}
          <span className="text-sm">{user.name}</span>
        </button>
      ) : (
        <button className="flex items-center gap-2 px-3 py-2 rounded-xl bg-white/10 hover:bg-white/20 border border-white/20 text-white" onClick={openLogin}>
          <LogIn className="w-5 h-5" />
          <span className="text-sm">Login</span>
        </button>
      )}
      {open && user && (
        <div className="absolute right-0 mt-2 w-44 bg-white rounded-xl shadow-lg border border-slate-200 overflow-hidden">
          <button className="w-full text-left px-4 py-2 text-sm hover:bg-slate-50">Profile</button>
          <button className="w-full text-left px-4 py-2 text-sm hover:bg-slate-50" onClick={logout}>
            <span className="inline-flex items-center gap-2 text-rose-600">
              <LogOut className="w-4 h-4" /> Logout
            </span>
          </button>
        </div>
      )}
      {/* Mount Login modal globally */}
      <LoginModal />
    </div>
  );
}
