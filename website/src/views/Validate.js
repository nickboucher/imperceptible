import {useState} from "react";

// core components
import IndexNavbar from "components/Navbars/IndexNavbar.js";
import Footer from "components/Footer/Footer.js";

// reactstrap components
import {
  Container,
  Row,
  Col,
  Form,
  FormGroup,
  Label,
  Input
} from "reactstrap";

export default function Validate() {
  const [input, setInput] = useState("");
  return (
    <>
      <IndexNavbar />
      <div className="wrapper">
        <div className="page-header d-flex flex-wrap">
          <img
            alt="..."
            className="dots"
            src={require("assets/img/dots.png").default}
          />
          <img
            alt="..."
            className="path"
            src={require("assets/img/path2.png").default}
          />
          <Container className="below-nav">
              <Row>
                  <Col md={12} className="text-center">
                    <h1><b>Validate</b></h1>
                  </Col>
              </Row>
              <Form>
                <FormGroup row className="mt-5 mb-n2">
                    <Label for="inputText" md={2}><h2>Input&nbsp;={'>'}</h2></Label>
                    <Col md={10}>
                        <Input
                            type="textarea"
                            name="inputText"
                            id="inputText"
                            placeholder="Paste some text here to validate whether it has been manipulated with imperceptible perturbations..."
                            onChange={e => setInput(e.target.value)}
                        />
                    </Col>
                </FormGroup>
            </Form>
            <Row className="mt-4 pt-4 mb-4 pb-4" style={{ display: input.length ? null : 'None' }}>
                <Col md={2}>
                    <h3>Output&nbsp;={'>'}</h3>
                </Col>
                <Col md={10}>
                    <div className="generated">
                        Not Yet Implemented.
                    </div>
                </Col>
            </Row>
          </Container>
          <Container className="pt-0 pb-4 align-self-end">
            <Row>
                <Col md={{ size: 10, offset: 2}}>
                  <p className="text-muted">This tool tests whether the input string contains encodings that may be indicators of imperceptible perturbations. It is not guaranteed to detect all forms of imperceptible perturbations. All text entered remains on your local machine. Nothing is transmitted to or logged on any server. This tool is for academic purposes only and the user holds sole responsibility for how it is used.</p>
                </Col>
            </Row>
        </Container>
      </div>
      <Footer />
    </div>
    </>
  );
}
