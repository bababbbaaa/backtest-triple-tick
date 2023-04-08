import React from "react";

interface BacktestButtonProps {
  onClick: () => void;
}

const BacktestButton: React.FC<BacktestButtonProps> = ({ onClick }) => {
  return (
    <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded" onClick={onClick}>
      Run Backtest
    </button>
  );
};

export default BacktestButton;
