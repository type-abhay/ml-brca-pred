import { useState, useEffect } from 'react';

function App() {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const [geneInfo, setGeneInfo] = useState([]);
  const [snackbarData, setSnackbarData] = useState(null);
  const [isSidePanelOpen, setIsSidePanelOpen] = useState(false);

  useEffect(() => {
    fetch('http://localhost:8000/info')
      .then(res => res.json())
      .then(data => setGeneInfo(data.gene_names))
      .catch(err => console.error("Failed to fetch gene info:", err));
  }, []);

  const handleDemo = async () => {
    setLoading(true); setError(null); setResults(null); setSnackbarData(null);
    try {
      const res = await fetch('http://localhost:8000/predict/demo');
      const data = await res.json();
      setResults(data);
      setSnackbarData(data.input_data);
      setIsSidePanelOpen(true);
    } catch (err) {
      setError("Failed to connect to the diagnostic engine.");
    }
    setLoading(false);
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setLoading(true); setError(null); setResults(null); setSnackbarData(null);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('http://localhost:8000/predict/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (res.ok) setResults(data);
      else setError(data.detail);
    } catch (err) {
      setError("Failed to upload the clinical data.");
    }
    setLoading(false);
  };

  // ⚠️ NEW FEATURE: Generate and download a blank CSV template
  const downloadTemplate = () => {
    // Creates a single row of comma-separated gene names
    const csvContent = geneInfo.join(',');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    
    // Programmatically trigger a hidden download link
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', 'brca_diagnostic_template.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-6 relative overflow-hidden">
      
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-[600px] h-[600px] bg-white/5 rounded-full blur-[120px] pointer-events-none"></div>

      <div className="z-10 w-full max-w-2xl">
        <h1 className="font-editorial text-5xl md:text-6xl text-center mb-2 tracking-tight">
          Omniscient <span className="text-white/50">BRCA</span>
        </h1>
        <p className="text-center text-white/40 mb-10 text-sm tracking-widest uppercase">
          Clinical Diagnostic Terminal
        </p>

        {/* Control Panel */}
        <div className="glass-panel rounded-3xl p-8 mb-8 flex flex-col gap-6 shadow-2xl relative">
          
          <div className="flex flex-col md:flex-row items-center gap-4 w-full">
            
            {/* Upload Button - Removed italics/font-editorial, using crisp sans-serif */}
            <label className="flex-1 w-full cursor-pointer group">
              <div className="h-16 flex items-center justify-center border-2 border-dashed border-white/20 rounded-xl group-hover:border-white/50 group-hover:bg-white/5 transition-all duration-300">
                <span className="text-lg font-semibold tracking-wide text-white/90 group-hover:text-white">Upload CSV Array</span>
              </div>
              <input type="file" accept=".csv" className="hidden" onChange={handleFileUpload} />
            </label>

            {/* Demo Button - Removed italics/font-editorial, using crisp sans-serif */}
            <button 
              onClick={handleDemo}
              className="flex-1 w-full h-16 bg-white text-black rounded-xl hover:bg-gray-200 transition-colors duration-300 shadow-[0_0_20px_rgba(255,255,255,0.2)] text-lg font-semibold tracking-wide"
            >
              Run Protocol: Demo
            </button>

            {/* Terminal Logs Trigger Button */}
            <button 
              onClick={() => setIsSidePanelOpen(true)}
              className="w-14 h-14 shrink-0 flex items-center justify-center rounded-full border border-white/20 hover:bg-white/10 transition-colors duration-300 group"
              title="View Required Genes & Logs"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.7)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="group-hover:stroke-white transition-colors">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="12" y1="16" x2="12" y2="12"></line>
                <line x1="12" y1="8" x2="12.01" y2="8"></line>
              </svg>
            </button>

          </div>
          
          {error && <div className="text-red-400 text-sm text-center border border-red-500/20 bg-red-500/10 p-3 rounded-lg">{error}</div>}
        </div>

        {/* Loading State */}
        {loading && (
          <div className="text-center text-white/50 animate-pulse font-editorial text-2xl">
            Sequencing RNA...
          </div>
        )}

        {/* Results Dashboard */}
        {results && (
          <div className="glass-panel rounded-3xl p-8 transition-opacity duration-500 opacity-100">
            <h2 className="text-sm tracking-widest uppercase text-white/40 mb-2">Final Diagnosis</h2>
            <div className="font-editorial text-4xl mb-8 border-b border-white/10 pb-6">
              {results.diagnosis.replace('_', ' ')}
            </div>

            <div className="space-y-5">
              {results.breakdown.map((item, index) => (
                <div key={item.subtype} className="w-full">
                  <div className="flex justify-between text-sm mb-2">
                    <span className={index === 0 ? "font-bold font-editorial text-xl" : "text-white/60 font-editorial text-lg"}>
                      {item.subtype.replace('BRCA_', '')}
                    </span>
                    <span className={index === 0 ? "font-bold" : "text-white/60 font-mono"}>
                      {item.confidence.toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-1.5 w-full bg-white/10 rounded-full overflow-hidden">
                    <div 
                      className={`h-full rounded-full transition-all duration-1000 ease-out ${index === 0 ? 'bg-white shadow-[0_0_10px_rgba(255,255,255,0.8)]' : 'bg-white/30'}`}
                      style={{ width: `${item.confidence}%` }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* THE SLIDING RIGHT PANEL */}
      {isSidePanelOpen && (
        <div 
          className="fixed inset-0 z-40" 
          onClick={() => setIsSidePanelOpen(false)}
        ></div>
      )}

      <div 
        className={`fixed top-0 right-0 h-full w-80 md:w-96 glass-panel border-l border-white/10 shadow-2xl z-50 transform transition-transform duration-500 ease-[cubic-bezier(0.16,1,0.3,1)] flex flex-col ${isSidePanelOpen ? 'translate-x-0' : 'translate-x-full'}`}
      >
        <div className="p-6 border-b border-white/10 flex justify-between items-center">
          <h3 className="font-editorial text-3xl tracking-tight text-white">Terminal Logs</h3>
          <button 
            onClick={() => setIsSidePanelOpen(false)}
            className="text-white/50 hover:text-white transition-colors"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto custom-scrollbar p-6 space-y-8">
          
          {/* Visual Bug Fixed: Removed the CSS animation that caused the leftward shift */}
          {snackbarData && (
            <div>
              <h4 className="text-[11px] text-white/50 uppercase tracking-widest mb-3 font-bold">Generated Demo Sequence</h4>
              <div className="bg-black/40 p-4 rounded-xl font-mono text-xs text-white/80 break-words leading-relaxed border border-white/5 shadow-inner">
                [{snackbarData.map(val => val.toFixed(4)).join(', ')}]
              </div>
            </div>
          )}

          <div>
            <h4 className="text-[11px] text-white/50 uppercase tracking-widest mb-2 font-bold">Required Panel ({geneInfo.length} Genes)</h4>
            <p className="text-xs text-white/40 mb-4 leading-relaxed">
              CSV must contain 1 row of Log2-transformed, scaled float values in this exact sequence:
            </p>
            <div className="bg-black/40 p-4 rounded-xl font-mono text-xs text-white/80 break-words leading-relaxed border border-white/5 shadow-inner mb-4">
              {geneInfo.join(', ')}
            </div>

            {/* ⚠️ NEW FEATURE: Download Template Button */}
            <button 
              onClick={downloadTemplate}
              className="w-full flex items-center justify-center gap-2 py-3 bg-white/10 hover:bg-white/20 border border-white/10 rounded-xl transition-colors duration-300 text-sm font-semibold tracking-wide"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="7 10 12 15 17 10"></polyline>
                <line x1="12" y1="15" x2="12" y2="3"></line>
              </svg>
              Download Blank CSV Template
            </button>

          </div>
        </div>
      </div>
    </div>
  );
}

export default App;