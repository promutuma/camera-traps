import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
  /** Optional label shown in the error card header, e.g. "Results" */
  label?: string;
}

interface State {
  error: Error | null;
}

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("[ErrorBoundary]", this.props.label ?? "page", error, info);
  }

  private reset = () => this.setState({ error: null });

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;

    const label = this.props.label ? `${this.props.label} — ` : "";

    return (
      <div className="flex items-center justify-center min-h-[40vh] p-8">
        <div className="max-w-lg w-full bg-red-50 border border-red-200 rounded-xl p-6 space-y-4">
          <div className="flex items-center gap-3">
            <span className="text-red-500 text-2xl">⚠</span>
            <h2 className="text-lg font-semibold text-red-800">
              {label}Something went wrong
            </h2>
          </div>
          <p className="text-sm text-red-700 font-mono break-words bg-red-100 rounded px-3 py-2">
            {error.message}
          </p>
          <button
            onClick={this.reset}
            className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white text-sm font-medium rounded-lg"
          >
            Try again
          </button>
        </div>
      </div>
    );
  }
}
