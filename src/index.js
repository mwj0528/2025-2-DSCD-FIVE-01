import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App'; // App.jsx를 불러오는 부분

const root = ReactDOM.createRoot(document.getElementById('root'));

//
root.render(
  <React.StrictMode>
    <App /> 
  </React.StrictMode>
);