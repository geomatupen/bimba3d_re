import { useState, useEffect, useRef } from "react";
import { Upload, X, ZoomIn, ZoomOut, RotateCw, Download, ChevronLeft, ChevronRight, Image as ImageIcon } from "lucide-react";
import { api } from "../../api/client";
import UploadModal from "../UploadModal";

interface ImagesTabProps {
  projectId: string;
  onUploaded?: () => void;
}

interface ImageFile {
  name: string;
  url: string;
  thumbnailUrl: string;
}

export default function ImagesTab({ projectId, onUploaded }: ImagesTabProps) {
  const [images, setImages] = useState<ImageFile[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedImage, setSelectedImage] = useState<number | null>(null);
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [thumbLoaded, setThumbLoaded] = useState<Set<number>>(new Set());
  const [thumbError, setThumbError] = useState<Set<number>>(new Set());
  const [fullImageLoaded, setFullImageLoaded] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const imageRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    fetchImages();
  }, [projectId]);

  const fetchImages = async () => {
    try {
      const res = await api.get(`/projects/${projectId}/files`);
      const imageFiles = res.data.files?.images || [];
      setImages(imageFiles.map((img: { name: string }) => ({
        name: img.name,
        url: `${api.defaults.baseURL}/projects/${projectId}/image/${img.name}`,
        thumbnailUrl: `${api.defaults.baseURL}/projects/${projectId}/thumbnail/${img.name}`
      })));
    } catch (err) {
      console.error("Failed to fetch images:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleThumbLoad = (index: number) => {
    setThumbLoaded((prev) => new Set(prev).add(index));
  };

  const handleThumbError = (index: number) => {
    setThumbError((prev) => new Set(prev).add(index));
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || []);
    if (selectedFiles.length === 0) return;

    setError(null);

    try {
      const formData = new FormData();
      selectedFiles.forEach((file) => formData.append("images", file));
      await api.post(`/projects/${projectId}/images`, formData);
      await fetchImages(); // Refresh the image list
      onUploaded?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    }
  };

  const openModal = (index: number) => {
    setSelectedImage(index);
    setFullImageLoaded(false);
    setZoom(1);
    setPosition({ x: 0, y: 0 });
  };

  const closeModal = () => {
    setSelectedImage(null);
    setFullImageLoaded(false);
    setZoom(1);
    setPosition({ x: 0, y: 0 });
  };

  const handleZoomIn = () => setZoom(prev => Math.min(prev + 0.5, 5));
  const handleZoomOut = () => setZoom(prev => Math.max(prev - 0.5, 0.5));
  const handleResetZoom = () => {
    setZoom(1);
    setPosition({ x: 0, y: 0 });
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (zoom > 1) {
      setIsDragging(true);
      setDragStart({ x: e.clientX - position.x, y: e.clientY - position.y });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging && zoom > 1) {
      setPosition({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y
      });
    }
  };

  const handleMouseUp = () => setIsDragging(false);

  const handlePrevious = () => {
    if (selectedImage !== null && selectedImage > 0) {
      setSelectedImage(selectedImage - 1);
      setFullImageLoaded(false);
      setZoom(1);
      setPosition({ x: 0, y: 0 });
    }
  };

  const handleNext = () => {
    if (selectedImage !== null && selectedImage < images.length - 1) {
      setSelectedImage(selectedImage + 1);
      setFullImageLoaded(false);
      setZoom(1);
      setPosition({ x: 0, y: 0 });
    }
  };

  const handleDownload = () => {
    if (selectedImage !== null) {
      const link = document.createElement('a');
      link.href = images[selectedImage].url;
      link.download = images[selectedImage].name;
      link.click();
    }
  };

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (selectedImage === null) return;
      
      if (e.key === 'Escape') closeModal();
      if (e.key === 'ArrowLeft') handlePrevious();
      if (e.key === 'ArrowRight') handleNext();
      if (e.key === '+' || e.key === '=') handleZoomIn();
      if (e.key === '-') handleZoomOut();
      if (e.key === '0') handleResetZoom();
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedImage, images.length]);

  return (
    <div className="max-w-7xl">
      {/* Upload Button */}
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-lg font-bold text-gray-900">
          Images ({images.length})
        </h2>
        <button
          onClick={() => setIsUploadOpen(true)}
          className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg transition-colors shadow-md"
        >
          <Upload className="w-4 h-4" />
          Upload Images
        </button>
      </div>

      {error && (
        <div className="mb-4 bg-red-50 border border-red-200 text-red-800 px-4 py-3 rounded-lg text-sm">
          {error}
        </div>
      )}

      {/* Images Grid */}
      {loading ? (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
          {[...Array(12)].map((_, i) => (
            <div key={i} className="aspect-square bg-slate-200 rounded-lg animate-pulse" />
          ))}
        </div>
      ) : images.length === 0 ? (
        <div className="text-center py-16 bg-white border-2 border-dashed border-slate-300 rounded-xl">
          <Upload className="w-12 h-12 text-slate-400 mx-auto mb-3" />
          <p className="text-sm text-slate-600 mb-4">No images uploaded yet</p>
          <label className="inline-flex items-center gap-2 px-5 py-2 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg cursor-pointer transition-colors shadow-md">
            <Upload className="w-4 h-4" />
            Upload Your First Images
            <input
              type="file"
              multiple
              accept="image/*"
              onChange={handleFileChange}
              className="hidden"
            />
          </label>
        </div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
          {images.map((image, index) => (
            <button
              key={index}
              onClick={() => openModal(index)}
              className="aspect-square rounded-lg overflow-hidden bg-slate-100 hover:ring-2 hover:ring-blue-500 transition-all group relative"
            >
              {/* Thumbnail skeleton */}
              {!thumbLoaded.has(index) && !thumbError.has(index) && (
                <div className="absolute inset-0 bg-slate-200 animate-pulse" />
              )}
              {/* Thumbnail or placeholder */}
              {thumbError.has(index) ? (
                <div className="w-full h-full flex items-center justify-center bg-slate-100 text-slate-500">
                  <ImageIcon className="w-6 h-6" />
                </div>
              ) : (
                <img
                  src={image.thumbnailUrl}
                  alt={image.name}
                  loading="lazy"
                  onLoad={() => handleThumbLoad(index)}
                  onError={() => handleThumbError(index)}
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-200"
                />
              )}
              <div className="absolute inset-0 bg-black/0 group-hover:bg-black/30 transition-colors flex items-center justify-center">
                <ZoomIn className="w-6 h-6 text-white opacity-0 group-hover:opacity-100 transition-opacity" />
              </div>
            </button>
          ))}
        </div>
      )}

      {/* Image Modal */}
      {selectedImage !== null && (
        <div className="fixed inset-0 bg-black/90 z-50 flex items-center justify-center">
          {/* Controls Bar */}
          <div className="absolute top-0 left-0 right-0 bg-black/50 backdrop-blur-sm p-4 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-white text-sm font-medium">
                {selectedImage + 1} / {images.length}
              </span>
              <span className="text-slate-300 text-sm">
                {images[selectedImage].name}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={handleZoomOut}
                className="p-2 hover:bg-white/20 rounded-lg transition-colors"
                title="Zoom Out (-)"
              >
                <ZoomOut className="w-5 h-5 text-white" />
              </button>
              <button
                onClick={handleResetZoom}
                className="p-2 hover:bg-white/20 rounded-lg transition-colors"
                title="Reset (0)"
              >
                <RotateCw className="w-5 h-5 text-white" />
              </button>
              <button
                onClick={handleZoomIn}
                className="p-2 hover:bg-white/20 rounded-lg transition-colors"
                title="Zoom In (+)"
              >
                <ZoomIn className="w-5 h-5 text-white" />
              </button>
              <span className="text-white text-sm font-medium px-2">
                {(zoom * 100).toFixed(0)}%
              </span>
              <button
                onClick={handleDownload}
                className="p-2 hover:bg-white/20 rounded-lg transition-colors"
                title="Download"
              >
                <Download className="w-5 h-5 text-white" />
              </button>
              <button
                onClick={closeModal}
                className="p-2 hover:bg-white/20 rounded-lg transition-colors"
                title="Close (Esc)"
              >
                <X className="w-5 h-5 text-white" />
              </button>
            </div>
          </div>

          {/* Navigation Arrows */}
          {selectedImage > 0 && (
            <button
              onClick={handlePrevious}
              className="absolute left-4 top-1/2 -translate-y-1/2 p-3 bg-black/50 hover:bg-black/70 rounded-full transition-colors"
              title="Previous (←)"
            >
              <ChevronLeft className="w-6 h-6 text-white" />
            </button>
          )}
          {selectedImage < images.length - 1 && (
            <button
              onClick={handleNext}
              className="absolute right-4 top-1/2 -translate-y-1/2 p-3 bg-black/50 hover:bg-black/70 rounded-full transition-colors"
              title="Next (→)"
            >
              <ChevronRight className="w-6 h-6 text-white" />
            </button>
          )}

          {/* Image Container */}
          <div
            className="w-full h-full flex items-center justify-center p-20 cursor-move"
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
          >
            {!fullImageLoaded && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="flex flex-col items-center gap-3">
                  <div className="w-8 h-8 border-4 border-blue-500 border-t-white rounded-full animate-spin" />
                  <p className="text-white text-sm">Loading image...</p>
                </div>
              </div>
            )}
            <img
              ref={imageRef}
              src={images[selectedImage].url}
              alt={images[selectedImage].name}
              onLoad={() => setFullImageLoaded(true)}
              className={`max-w-full max-h-full object-contain select-none ${!fullImageLoaded ? 'opacity-0' : 'opacity-100'} transition-opacity duration-300`}
              style={{
                transform: `scale(${zoom}) translate(${position.x / zoom}px, ${position.y / zoom}px)`,
                cursor: zoom > 1 ? (isDragging ? 'grabbing' : 'grab') : 'default',
                transition: isDragging ? 'none' : 'transform 0.2s ease-out'
              }}
              draggable={false}
            />
          </div>
        </div>
      )}

      {/* Upload Modal Mount */}
      <UploadModal
        projectId={projectId}
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
        onUploaded={async () => {
          await fetchImages();
        }}
      />
    </div>
  );
}
