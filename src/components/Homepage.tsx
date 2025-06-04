import React from "react";

const Homepage: React.FC = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-400 via-purple-500 to-pink-500 flex items-center justify-center">
      <div className="bg-white rounded-lg shadow-xl p-8 text-center">
        <h1 className="text-4xl font-bold text-gray-800 mb-4">Hello World</h1>
        <p className="text-gray-600 text-lg">Welcome to Action Recognition</p>
      </div>
    </div>
  );
};

export default Homepage;
