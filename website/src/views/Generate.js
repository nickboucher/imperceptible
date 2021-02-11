import {useState, useEffect} from "react";

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
    Input,
    Button,
    Tooltip,
    CustomInput
  } from "reactstrap";
  import {CopyToClipboard} from 'react-copy-to-clipboard';

const TooltipContent = ({ scheduleUpdate, copied }) => {
    useEffect(() => scheduleUpdate(), [copied])
    return (
      <>{ copied ? "Copied!" : "Copy to Clipboard" }</>
    );
  }
  

export default function Generate() {
    const [output, setOutput] = useState("");
    const [copied, setCopied] = useState(false);
    const [tooltipOpen, setTooltipOpen] = useState(false);
    const toggle = () => {
      setTooltipOpen(!tooltipOpen);
      setCopied(false);
    };
    useEffect(() => {
        document.body.classList.toggle("profile-page");
        // Specify how to clean up after this effect:
        return function cleanup() {
          document.body.classList.toggle("profile-page");
        };
      },[]);
  return (
    <>
      <IndexNavbar />
      <div className="wrapper">
        <div className="page-header">
          <img
            alt="..."
            className="dots"
            src={require("assets/img/dots.png").default}
          />
          <img
            alt="..."
            className="path"
            src={require("assets/img/path4.png").default}
          />
          <Container className="below-nav">
              <Row>
                  <Col md={12} className="text-center">
                    <h1><b>Generate</b></h1>
                  </Col>
              </Row>
            <Form>
                <FormGroup row className="mt-5">
                    <Label for="inputText" md={2}><h2>Input ={'>'}</h2></Label>
                    <Col md={10}>
                        <Input
                            type="textarea"
                            name="inputText"
                            id="inputText"
                            placeholder="Type some input text to generate a random imperceptibly perturbed output..."
                            onChange={e => setOutput(e.target.value)}
                        />
                    </Col>
                </FormGroup>
                <FormGroup row>
                  <Col md={{ size: 10, offset: 2}}>
                    <CustomInput type="switch" id="invisibles" label="Invisible Characters" defaultChecked={true} inline={true} />
                    <CustomInput type="switch" id="homoglyphs" label="Homoglyphs" defaultChecked={true} inline={true} />
                    <CustomInput type="switch" id="reorderings" label="Reorderings" defaultChecked={true} inline={true} />
                    <CustomInput type="switch" id="deletions" label="Deletions" defaultChecked={true} inline={true} />
                  </Col>
                </FormGroup>
            </Form>
            <Row className="mt-4 pt-4" style={{ display: output.length ? null : 'None' }}>
                <Col md={2}>
                    <h3>Output ={'>'}</h3>
                </Col>
                <Col md={10}>
                    <div className="generated">
                        <Row>
                          <Col lg={10} sm={9}>
                            <p>{output}</p>
                          </Col>
                          <Col lg={2} sm={3}>
                            <CopyToClipboard text={output} onCopy={(text,result) => setCopied(result) }>
                              <Button id="copy"><i className="far fa-copy"></i></Button>
                            </CopyToClipboard>
                            <Tooltip delay={0} target="copy" isOpen={tooltipOpen} toggle={toggle}>
                              {({scheduleUpdate}) => (<TooltipContent copied={copied} scheduleUpdate={scheduleUpdate} />)}
                            </Tooltip>
                          </Col>
                        </Row>
                    </div>
                </Col>
            </Row>
        </Container>
      </div>
      <Footer />
    </div>
    </>
  );
}
