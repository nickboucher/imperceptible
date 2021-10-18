import { useState, useEffect } from "react";
import useDimensions from "react-use-dimensions";
import { useWindowHeight } from '@react-hook/window-size';

// core components
import IndexNavbar from "components/IndexNavbar.js";
import Footer from "components/Footer.js";

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

import {invisibleChars, homoglyphChars} from "variables/Constants.js";

// Generates a random integer in (min,max]
const randrange = (min, max) => {
    return Math.floor(Math.random() * (max - min) ) + min;
  }

// Generates a random integer in (0,max]
const rand = (max) => randrange(0, max);

// Generates a random invisible character
const invisible = () => {
  return invisibleChars[rand(invisibleChars.length)];
}

const TooltipContent = ({ scheduleUpdate, copied }) => {
    useEffect(scheduleUpdate, [copied, scheduleUpdate])
    return (
      <>{ copied ? "Copied!" : "Copy to Clipboard" }</>
    );
  }

export default function Generate() {
    const [copied, setCopied] = useState(false);
    const [tooltipOpen, setTooltipOpen] = useState(false);
    const [input, setInput] = useState("");
    const [output, setOutput] = useState("");
    const [outputHtml, setOutputHtml] = useState("");
    const [invisibles, setInvisibles] = useState(true);
    const [homoglyphs, setHomoglyphs] = useState(true);
    const [reorderings, setReorderings] = useState(true);
    const [deletions, setDeletions] = useState(false);
    const [footerRef, { height: footerHeight }] = useDimensions();
    const windowHeight = useWindowHeight();
    const [minHeight, setMinHeight] = useState('100vh');
    useEffect(() => {
      if (footerHeight && windowHeight) {
        setMinHeight(windowHeight - footerHeight);
      } else {
        setMinHeight('100vh');
      }
    }, [footerHeight, windowHeight, setMinHeight]);
    useEffect(() => {
      let results = input.split("\n");
      for (let r=0; r<results.length; r++) {
        let result = results[r].split("");
        // Inject invisible characters
        if (invisibles) {
          const injections = randrange(Math.min(1,result.length), result.length);
          for (let i=0; i<injections; i++) {
            result.splice(rand(result.length), 0, invisible())
          }
        }
        // Inject homoglyphs
        if (homoglyphs) {
          const injections = randrange(Math.min(1,result.length), result.length);
          for (let i=0; i<injections; i++) {
            let j = rand(result.length);
            if (result[j] in homoglyphChars) {
              result[j] = homoglyphChars[result[j]];
            }
            else {
              // If the random character doens't have a homoglpyh, iterate through input
              // until we find a possible swap
              for (let k=0; k<result.length; k++) {
                if (result[k] in homoglyphChars) {
                  result[k] = homoglyphChars[result[k]];
                  break;
                }
              }
            }
          }
        }
        if (deletions) {
          const injections = randrange(Math.min(1,result.length), result.length);
          for (let i=0; i<injections; i++) {
            // Generate a single random non-control ASCII character to delete
            // Note: Most web browsers won't perform the deletion on rendering
            let del = `${String.fromCharCode(randrange(32,127))}\x08`;
            result.splice(rand(result.length), 0, del);
          }
        }
        if (reorderings) {
          // Note: algorithm targets Chromium as of 2021
          const injections = randrange(Math.min(1,result.length), result.length);
          for (let i=0; i<injections; i++) {
            if (result.length >= 2) {
              const index = rand(result.length-1);
              const swap = `\u202D\u2066\u202E\u2067${result[index+1]}\u2069\u2066${result[index]}\u2069\u202C\u2069\u202C`;
              result.splice(index, 2, swap);
            }
          }
        }
        results[r] = result.join("");
      }
      setOutput(results.join("\n"));
      setOutputHtml(results.map((result, index) => <p key={`p${index}`}>{result.replace(" ", "\xA0")}</p>));
    }, [input, invisibles, homoglyphs, reorderings, deletions]);
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
        <div className="page-header d-flex flex-wrap" style={{ minHeight: minHeight }}>
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
          <div className="w-100">
            <Container className="below-nav mb-4">
                <Row>
                    <Col md={12} className="text-center">
                      <h1 className="xxs"><b>Perturbation<br />Generator</b></h1>
                    </Col>
                </Row>
                <Row>
                  <Col md={12} className="text-center">
                    <h3>Type some text below to generate imperceptible perturbations.<sup style={{ fontSize: '.5em', top: '-1em' }}>&dagger;</sup></h3>
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
                              placeholder="Type some text to generate a random imperceptible perturbation..."
                              onChange={e => setInput(e.target.value)}
                          />
                      </Col>
                  </FormGroup>
                  <FormGroup row>
                    <Col md={{ size: 10, offset: 2}}>
                      <Row>
                        <Col lg={3} xs={6} className="mt-3">
                          <CustomInput type="switch" id="invisibles" label="Invisibles" checked={invisibles} onChange={e =>setInvisibles(e.target.checked)} inline={true} />
                        </Col>
                        <Col lg={3} xs={6} className="mt-3">
                          <CustomInput type="switch" id="homoglyphs" label="Homoglyphs" checked={homoglyphs} onChange={e =>setHomoglyphs(e.target.checked)} inline={true} />
                        </Col>
                        <Col lg={3} xs={6} className="mt-3">
                          <CustomInput type="switch" id="reorderings" label="Reorderings" checked={reorderings} onChange={e =>setReorderings(e.target.checked)} inline={true} />
                        </Col>
                        <Col lg={3} xs={6} className="mt-3">
                          <CustomInput type="switch" id="deletions" label="Deletions" checked={deletions} onChange={e =>setDeletions(e.target.checked)} inline={true} />
                        </Col>
                      </Row>
                    </Col>
                  </FormGroup>
              </Form>
              <Row className="mt-4 pt-4 mb-4 pb-4" style={{ display: output.length ? null : 'None' }}>
                  <Col md={2}>
                      <h3>Output&nbsp;={'>'}</h3>
                  </Col>
                  <Col md={10}>
                      <div className="generated">
                          <Row>
                            <Col lg={10} sm={9}>
                              {outputHtml}
                            </Col>
                            <Col lg={2} sm={3} className="text-right">
                              <CopyToClipboard text={output} onCopy={(text,result) => setCopied(result) }>
                                <Button id="copy" className="btn-flat"><i className="far fa-copy"></i></Button>
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
          <div className="w-100 align-self-end">
            <Container className="pt-4 pb-4">
              <Row>
                  <Col md={{ size: 10, offset: 2}}>
                    <p className="text-muted"><sup>&dagger;</sup> This tool generates a randomly chosen version of the input string with the selected imperceptible perturbations applied. All text entered remains on your local machine. Nothing is transmitted to or logged on any server. This tool is for academic purposes only and the user holds sole responsibility for how it is used.</p>
                    <p className="text-muted">Compatability: Invisible characters and homoglpyhs should be compatible with most pieces of software supporting Unicode, which includes modern broswers. The algorithm used to generate reorderings is Unicode rendering engine specific, and was written to target Chromium-based software. Reorderings have been tested to render correctly on Google Chrome, the new Microsoft Edge, Safari, and Eletron-based applications. Due to the way most browsers render text, deletions are unlikely to render correctly in any modern browser. These values may render correctly when pasted into other applictions, such as any system which renders text using Python's Unicode engine.</p>
                  </Col>
              </Row>
          </Container>
          </div>
      </div>
      <Footer ref={footerRef}/>
    </div>
    </>
  );
}
