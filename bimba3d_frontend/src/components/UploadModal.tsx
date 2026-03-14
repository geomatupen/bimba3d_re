import { useCallback, useState } from "react";
import { Upload, X, FileImage, Trash2 } from "lucide-react";
import { api } from "../api/client";

interface UploadModalProps {
  projectId: string;
  isOpen: boolean;
  onClose: () => void;
  onUploaded?: (count: number) => void;
}

export default function UploadModal({ projectId, isOpen, onClose, onUploaded }: UploadModalProps) {
  const [files, setFiles] = useState<File[]>([]);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(false);
    const dropped = Array.from(e.dataTransfer.files || []);
    if (dropped.length) setFiles((prev) => [...prev, ...dropped]);
  }, []);

  const onDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(true);
  }, []);

  const onDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(false);
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = Array.from(e.target.files || []);
    if (selected.length) setFiles((prev) => [...prev, ...selected]);
  };

  const removeFile = (idx: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== idx));
  };

  const handleUpload = async () => {
    if (!files.length) return;
    setUploading(true);
    setError(null);
    try {
      const formData = new FormData();
      files.forEach((file) => formData.append("images", file));
      const res = await api.post(`/projects/${projectId}/images`, formData);
      onUploaded?.(res.data?.count || files.length);
      setFiles([]);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50">
      <div className="absolute inset-0 bg-black/60" onClick={() => !uploading && onClose()} />
      <div className="absolute inset-0 flex items-center justify-center p-4">
        <div className="w-full max-w-2xl bg-white rounded-xl shadow-2xl border border-slate-200 overflow-hidden">
          {/* Header */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-slate-200">
            <h3 className="text-base font-semibold text-slate-900">Upload Images</h3>
            <button
              className="p-2 rounded-lg hover:bg-slate-100"
              onClick={() => !uploading && onClose()}
              aria-label="Close"
            >
              <X className="w-5 h-5 text-slate-700" />
            </button>
          </div>

          {/* Body */}
          <div className="px-4 py-4">
            {error && (
              <div className="mb-3 bg-rose-50 border border-rose-200 text-rose-700 px-3 py-2 rounded-lg text-sm">
                {error}
              </div>
            )}

            <div
              className={`rounded-xl border-2 border-dashed ${dragging ? 'border-blue-400 bg-blue-50' : 'border-slate-300 bg-slate-50'} p-6 flex flex-col items-center justify-center text-center transition-colors`}
              onDrop={onDrop}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
            >
              <Upload className="w-10 h-10 text-slate-400 mb-2" />
              <p className="text-sm text-slate-600 mb-3">
                Drag & drop images here, or click to browse
              </p>
              <label className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg cursor-pointer transition-colors">
                <Upload className="w-4 h-4" />
                Select Images
                <input type="file" multiple accept="image/*" className="hidden" onChange={handleFileChange} />
              </label>
            </div>

            {files.length > 0 && (
              <div className="mt-4">
                <h4 className="text-sm font-semibold text-slate-900 mb-2">Selected Files ({files.length})</h4>
                <ul className="space-y-2 max-h-40 overflow-auto">
                  {files.map((f, idx) => (
                    <li key={idx} className="flex items-center justify-between bg-slate-50 rounded-lg border border-slate-200 px-3 py-2">
                      <div className="flex items-center gap-2">
                        <FileImage className="w-4 h-4 text-slate-500" />
                        <span className="text-sm text-slate-700 truncate max-w-[14rem]">{f.name}</span>
                        <span className="text-xs text-slate-500">{(f.size / 1024 / 1024).toFixed(2)} MB</span>
                      </div>
                      <button className="p-2 rounded-lg hover:bg-slate-100" onClick={() => removeFile(idx)} aria-label="Remove">
                        <Trash2 className="w-4 h-4 text-slate-600" />
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="px-4 py-3 border-t border-slate-200 flex items-center justify-end gap-2">
            <button
              className="px-4 py-2 rounded-lg border border-slate-300 text-slate-700 hover:bg-slate-50"
              onClick={() => !uploading && onClose()}
            >
              Cancel
            </button>
            <button
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-600 text-white font-semibold hover:bg-blue-700 disabled:opacity-50"
              disabled={uploading || files.length === 0}
              onClick={handleUpload}
            >
              <Upload className="w-4 h-4" />
              {uploading ? "Uploading..." : "Upload"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
