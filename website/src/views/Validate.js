import React from "react";

// core components
import IndexNavbar from "components/Navbars/IndexNavbar.js";
import Footer from "components/Footer/Footer.js";

export default function Validate() {
  React.useEffect(() => {
    document.body.classList.toggle("validate-page");
    // Specify how to clean up after this effect:
    return function cleanup() {
      document.body.classList.toggle("validate-page");
    };
  },[]);
  return (
    <>
      <IndexNavbar />
      <div className="wrapper">
        <div className="main">
          <h1>Content</h1>
        </div>
        <Footer />
      </div>
    </>
  );
}
