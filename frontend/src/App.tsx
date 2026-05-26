import ChatPDF from "./app/chat-pdf";
import ListPDF from "./app/list-pdf";
import Navbar from "./app/navbar";
import ViewPDF from "./app/view-pdf";

export default function App() {
  return (
    <div className="h-screen flex flex-col overflow-hidden max-w-[1600px] mx-auto w-full px-4 lg:px-6">
      <Navbar />
      <main className="flex-grow flex items-stretch pt-2 pb-4 lg:pb-6 w-full overflow-hidden min-h-0">
        <div className="flex flex-col lg:grid lg:grid-cols-12 gap-6 w-full h-full">
          <div className="lg:col-span-3 h-full overflow-hidden">
            <ListPDF />
          </div>
          <div className="lg:col-span-9 flex flex-col lg:grid lg:grid-cols-2 gap-6 h-full overflow-hidden">
            <ViewPDF />
            <ChatPDF />
          </div>
        </div>
      </main>
    </div>
  );
}
