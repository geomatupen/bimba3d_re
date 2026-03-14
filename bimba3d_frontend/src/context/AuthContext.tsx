import { createContext, useContext, useEffect, useState } from "react";

type User = {
  id: string;
  name: string;
  email: string;
  avatarUrl?: string;
};

type AuthContextType = {
  user: User | null;
  loginWithGoogle: () => void;
  logout: () => void;
  openLogin: () => void;
  openSignup: () => void;
  closeModals: () => void;
  isLoginOpen: boolean;
  isSignupOpen: boolean;
};

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoginOpen, setIsLoginOpen] = useState(false);
  const [isSignupOpen, setIsSignupOpen] = useState(false);

  useEffect(() => {
    const raw = localStorage.getItem("bimba3d:user");
    if (raw) {
      try { setUser(JSON.parse(raw)); } catch {}
    }
  }, []);

  const loginWithGoogle = () => {
    // Stub: replace with backend OAuth later
    const demoUser: User = {
      id: "demo-1",
      name: "Demo User",
      email: "demo@example.com",
      avatarUrl: "https://ui-avatars.com/api/?name=Demo+User&background=0D8ABC&color=fff"
    };
    setUser(demoUser);
    localStorage.setItem("bimba3d:user", JSON.stringify(demoUser));
    setIsLoginOpen(false);
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem("bimba3d:user");
  };

  const openLogin = () => setIsLoginOpen(true);
  const openSignup = () => setIsSignupOpen(true);
  const closeModals = () => { setIsLoginOpen(false); setIsSignupOpen(false); };

  return (
    <AuthContext.Provider value={{ user, loginWithGoogle, logout, openLogin, openSignup, closeModals, isLoginOpen, isSignupOpen }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
