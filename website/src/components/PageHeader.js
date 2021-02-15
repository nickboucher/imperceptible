import React from "react";

// reactstrap components
import { Container } from "reactstrap";

export default function PageHeader() {
  return (
    <div className="page-header header-filter">
      <div className="squares square1"><span>𐀴</span></div>
      <div className="squares square2"><span>ฆ</span></div>
      <div className="squares square3"><span>ψ</span></div>
      <div className="squares square4"><span>A</span></div>
      <div className="squares square5"><span>Ж</span></div>
      <div className="squares square6"><span>水</span></div>
      <div className="squares square7"><span>ה</span></div>
      <Container>
        <div className="content-center brand">
          <h1 className="h1-seo">Imperceptible Perturbations</h1>
          <h3 className="d-none d-sm-block">
            A novel method to break text-based ML systems.
          </h3>
        </div>
      </Container>
    </div>
  );
}
