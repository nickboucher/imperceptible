import React from "react";
import { render } from 'react-snapshot'
import { BrowserRouter, Route, Switch, Redirect } from "react-router-dom";

import "assets/css/nucleo-icons.css";
import "assets/scss/blk-design-system-react.scss?v=1.2.0";
import "assets/demo/demo.css";

import Index from "views/Index.js";
import Validate from "views/Validate.js";
import Generate from "views/Generate.js";

render(
  <BrowserRouter>
    <Switch>
      <Route exact path="/" render={(props) => <Index {...props} />} />
      <Route path="/detector" render={(props) => <Validate {...props} />} />
      <Route path="/generator" render={(props) => <Generate {...props} />} />
      <Redirect to="/" />
    </Switch>
  </BrowserRouter>,
  document.getElementById("root")
);
