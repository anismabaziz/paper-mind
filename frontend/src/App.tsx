import ListPdf from "./app/list-pdf";
import Navbar from "./app/navbar";

export default function App() {
  return (
    <div>
      <Navbar />
      <div className="grid grid-cols-4 container mx-auto gap-4">
        <ListPdf />
      </div>
    </div>
  );
}
