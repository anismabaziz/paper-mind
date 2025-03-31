import ChatPDF from "./app/chat-pdf";
import ListPDF from "./app/list-pdf";
import Navbar from "./app/navbar";
import ViewPDF from "./app/view-pdf";

export default function App() {
  return (
    <div>
      <Navbar />
      <div className="grid grid-cols-4 container mx-auto gap-4">
        <ListPDF />
        <div className="grid col-span-3 grid-cols-2 gap-4">
          <ViewPDF />
          <ChatPDF />
        </div>
      </div>
    </div>
  );
}
