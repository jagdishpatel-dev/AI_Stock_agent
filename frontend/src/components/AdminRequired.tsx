import { Lock } from "lucide-react";
import { Link } from "react-router-dom";
import { EmptyState } from "./EmptyState";

export function AdminRequired({ what }: { what: string }) {
  return (
    <div className="panel">
      <div className="panel-body">
        <EmptyState
          icon={Lock}
          title="Admin key required"
          description={`${what} are private. Unlock the Admin page with your admin key to view them.`}
          action={<Link to="/admin">Go to Admin</Link>}
        />
      </div>
    </div>
  );
}
