module.exports = {
  content: ['./index.html', './app.js'],
  theme: {
    extend: {
      colors: { ink: '#172d2a', muted: '#78847f', forest: '#133d35', mint: '#edf5ef', paper: '#f7f9f7', line: '#e5ebe6' },
      fontFamily: { sans: ['Inter', 'Segoe UI', 'Arial', 'sans-serif'], display: ['Georgia', 'serif'] },
      boxShadow: { soft: '0 8px 40px -16px rgba(22,54,39,.14)' },
    },
  },
  plugins: [],
};
