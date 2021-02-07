import React from "react";

// core components
import IndexNavbar from "components/Navbars/IndexNavbar.js";
import PageHeader from "components/PageHeader/PageHeader.js";
import Footer from "components/Footer/Footer.js";

// sections for this page/view
import Basics from "views/ExampleSections/Basics.js";
import Navbars from "views/ExampleSections/Navbars.js";
import Tabs from "views/ExampleSections/Tabs.js";
import Pagination from "views/ExampleSections/Pagination.js";
import Notifications from "views/ExampleSections/Notifications.js";
import Typography from "views/ExampleSections/Typography.js";
import JavaScript from "views/ExampleSections/JavaScript.js";
import NucleoIcons from "views/ExampleSections/NucleoIcons.js";
import Signup from "views/ExampleSections/Signup.js";
import Examples from "views/ExampleSections/Examples.js";
import Download from "views/ExampleSections/Download.js";

export default function Examples() {
  React.useEffect(() => {
    document.body.classList.toggle("index-page");
    // Specify how to clean up after this effect:
    return function cleanup() {
      document.body.classList.toggle("index-page");
    };
  },[]);
  return (
    <>
      <IndexNavbar />
      <div className="wrapper">
        <PageHeader />
        <div className="main">
          <Basics />
          <Navbars />
          <Tabs />
          <Pagination />
          <Notifications />
          <Typography />
          <JavaScript />
          <NucleoIcons />
          <Signup />
          <Examples />
          <Download />
        </div>
        <Footer />
      </div>
    </>
  );
}
